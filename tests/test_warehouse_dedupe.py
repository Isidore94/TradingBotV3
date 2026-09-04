"""Partition de-duplication at the dataset grain (BD-96).

On 2026-09-03 ``bar_m5 month=2026-08`` held 12,015,283 rows for 1,816,970
distinct (symbol, interval_start) keys - 85% duplicates - because the tee's
dedupe state reset every UTC midnight and every restart and the seal trusted
it. The repair is a COMPACT-shaped rewrite: one manifest line, inputs retired
never deleted, the earliest observation kept, the drop counted on the line.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from scripts.research_warehouse import cli, schemas
from scripts.research_warehouse.schemas import DATASETS, SCHEMA_VERSION
from scripts.research_warehouse.store import ResearchStore

UTC = timezone.utc
OPEN = datetime(2026, 8, 3, 13, 30, tzinfo=UTC)
PARTITION = "month=2026-08"


def _row(minute: int, observed_minute: int, symbol: str = "AAPL") -> dict:
    start = OPEN + timedelta(minutes=minute)
    return {
        "symbol": symbol,
        "interval_start": start,
        "interval_end": start + timedelta(minutes=5),
        "session_id": "XNYS-2026-08-03",
        "session_phase": "RTH",
        # The open encodes WHICH observation this is, so the test can tell
        # the kept copy from the dropped one.
        "open": 100.0 + observed_minute,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000,
        "vwap": None,
        "trade_count": None,
        "provider": "IBKR",
        "is_complete": True,
        "quality": "COMPLETE",
        "source_hash": "abc",
        "event_at": start + timedelta(minutes=5),
        "observed_at": start + timedelta(minutes=observed_minute),
        "capture_mode": "LIVE",
        "revision_id": "",
        "supersedes_revision_id": "",
        "schema_version": SCHEMA_VERSION,
        "run_id": "cycle-1",
    }


@pytest.fixture()
def duplicated_store(tmp_path):
    store = ResearchStore.open(tmp_path / "lake")
    store.publish("bar_m5", [_row(0, 6), _row(5, 11)])
    # A later tick re-offers the same two bars and adds one new one.
    store.publish("bar_m5", [_row(0, 60), _row(5, 65), _row(10, 16)])
    return store


def test_the_dry_run_counts_and_writes_nothing(duplicated_store):
    store = duplicated_store
    before = store.manifest.path.read_bytes()
    result = store.duplicate_rows("bar_m5", PARTITION)
    assert (result.rows_before, result.rows_after, result.rows_dropped) == (5, 3, 2)
    assert result.entry is None
    assert store.manifest.path.read_bytes() == before
    assert store.read_table("bar_m5").num_rows == 5


def test_dedupe_keeps_the_earliest_observation_and_retires_the_inputs(duplicated_store):
    store = duplicated_store
    result = store.dedupe_partition("bar_m5", PARTITION, job_id="test-dedupe")

    assert (result.rows_before, result.rows_after, result.rows_dropped) == (5, 3, 2)
    table = store.read_table("bar_m5")
    assert table.num_rows == 3
    kept = {row["interval_start"]: row["open"] for row in table.to_pylist()}
    assert kept[OPEN] == 106.0 and kept[OPEN + timedelta(minutes=5)] == 111.0, "the first capture is the evidence"
    assert kept[OPEN + timedelta(minutes=10)] == 116.0

    entry = result.entry
    assert entry is not None and entry.action == "COMPACT"
    assert len(entry.supersedes) == 2
    assert entry.extra["rows_dropped"] == 2
    assert entry.extra["dedupe_grain"] == ["symbol", "interval_start", "provider", "revision_id"]
    assert set(store.retired_pending()) == set(entry.supersedes), "inputs are retired, never deleted"

    # Idempotent: a clean partition gets no new line and no new file.
    ledger = store.manifest.path.read_bytes()
    again = store.dedupe_partition("bar_m5", PARTITION, job_id="test-dedupe")
    assert again.rows_dropped == 0 and again.entry is None
    assert store.manifest.path.read_bytes() == ledger


def test_dedupe_refuses_superseding_and_never_compactable_datasets(tmp_path, monkeypatch):
    store = ResearchStore.open(tmp_path / "lake")
    with pytest.raises(ValueError, match="normal shape"):
        store.dedupe_partition("outcome_path", "year=2026")
    spec = DATASETS["bar_m5"]
    frozen = schemas.DatasetSpec(
        name=spec.name,
        layer=spec.layer,
        schema=spec.schema,
        time_column=spec.time_column,
        partition_by=spec.partition_by,
        grain=spec.grain,
        compactable=False,
    )
    monkeypatch.setitem(DATASETS, "bar_m5", frozen)
    with pytest.raises(ValueError, match="never a compaction input"):
        store.dedupe_partition("bar_m5", PARTITION)


def test_the_cli_dedupe_is_a_dry_run_unless_applied(duplicated_store):
    store = duplicated_store
    report = cli.run_dedupe(store, dataset="bar_m5")
    assert report["applied"] is False and report["rows_dropped"] == 2
    assert report["partitions"] == [
        {"partition": PARTITION, "rows_before": 5, "rows_after": 3, "rows_dropped": 2, "rewritten": False}
    ]
    assert store.read_table("bar_m5").num_rows == 5

    applied = cli.run_dedupe(store, dataset="bar_m5", apply=True)
    assert applied["applied"] is True and applied["rows_dropped"] == 2
    assert applied["partitions"][0]["rewritten"] is True
    assert store.read_table("bar_m5").num_rows == 3


def test_the_cli_dedupe_is_inert_without_a_store():
    assert cli.run_dedupe(None)["status"] == "DISABLED"
