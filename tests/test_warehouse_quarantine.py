"""Partial-publish semantics and the tracker-incident regression (plan sec 8.3).

The precedent this pins is the tracker blackout of the week of 2026-07-13: a
bounded dirty tail vetoed an entire publish, so days of capture were lost
because a handful of records were malformed. The lake must never repeat it.

Contract (sec 8.3, risk R4):
- dirty records are quarantined at per-symbol / per-partition granularity;
- the clean remainder publishes in the same call;
- the quarantine count surfaces in Health;
- malformed records are moved, never silently discarded;
- only manifest corruption vetoes a publish wholesale.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.research_warehouse import schemas, store as store_mod
from scripts.research_warehouse.store import ResearchStore

UTC = timezone.utc


def _bar(symbol="AAPL", minute=0, **overrides):
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
        "volume": 1000,
        "vwap": 100.2,
        "trade_count": 10,
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


def test_dirty_tail_is_quarantined_and_the_clean_remainder_publishes(store):
    """The tracker-incident regression, stated as one call."""
    rows = [_bar("AAPL", 0), _bar("MSFT", 5), _bar("NVDA", 10)]
    # NVDA's tail is the dirt: a text price the provider never should have sent.
    rows.append(_bar("NVDA", 15, close="not-a-price"))
    rows.append(_bar("NVDA", 20, volume="lots"))

    result = store.publish("bar_m5", rows, job_id="tee")

    assert result.rows_published == 3
    assert result.rows_quarantined == 2
    assert store.read_table("bar_m5").num_rows == 3
    symbols = set(store.read_table("bar_m5").column("symbol").to_pylist())
    assert symbols == {"AAPL", "MSFT", "NVDA"}  # the clean NVDA bar still ships

    quarantined = store.manifest.quarantine_entries()
    assert len(quarantined) == 1
    assert quarantined[0].extra["symbol"] == "NVDA"
    assert quarantined[0].partition == "month=2026-08"
    assert quarantined[0].row_count == 2

    lines = (store.root / quarantined[0].file_path).read_text(encoding="utf-8").strip().splitlines()
    payloads = [json.loads(line) for line in lines]
    assert {p["row"]["close"] for p in payloads} == {"not-a-price", 100.5}
    assert all(p["reason"].startswith(store_mod.QUARANTINE_SCHEMA_CAST) for p in payloads)


def test_quarantine_granularity_is_per_symbol_and_partition(store):
    rows = [
        _bar("AAPL", 0, close="bad"),
        _bar("MSFT", 0, close="bad"),
        _bar("AAPL", 0, interval_start=datetime(2026, 9, 1, 13, 30, tzinfo=UTC), close="bad"),
        _bar("TSLA", 0),
    ]
    result = store.publish("bar_m5", rows)

    assert result.rows_published == 1
    keys = {(entry.partition, entry.extra["symbol"]) for entry in result.quarantined}
    assert keys == {
        ("month=2026-08", "AAPL"),
        ("month=2026-08", "MSFT"),
        ("month=2026-09", "AAPL"),
    }


def test_naive_timestamps_are_quarantined_never_guessed(store):
    naive = datetime(2026, 8, 3, 13, 30)
    result = store.publish("bar_m5", [_bar("AAPL", 0), _bar("MSFT", 0, observed_at=naive)])

    assert result.rows_published == 1 and result.rows_quarantined == 1
    reasons = result.quarantined[0].extra["reasons"]
    assert reasons == [store_mod.QUARANTINE_NAIVE_TIMESTAMP]


def test_unpartitionable_rows_are_quarantined_not_dropped(store):
    result = store.publish("bar_m5", [_bar("AAPL", 0), _bar("MSFT", 0, interval_start=None)])

    assert result.rows_published == 1 and result.rows_quarantined == 1
    entry = result.quarantined[0]
    assert entry.partition == "unpartitioned"
    assert entry.extra["reasons"] == [store_mod.QUARANTINE_PARTITION_KEY]
    payload = json.loads((store.root / entry.file_path).read_text(encoding="utf-8").strip())
    assert payload["row"]["symbol"] == "MSFT"


def test_domain_validator_rejections_are_quarantined_with_their_reason(store):
    def completed_bars_only(row):
        # Completed bars only for state transitions (plan.md sec 5); a forming
        # bar is preview evidence and never enters a completed-bar dataset.
        return None if row.get("is_complete") else "forming bar"

    result = store.publish(
        "bar_m5",
        [_bar("AAPL", 0), _bar("AAPL", 5, is_complete=False)],
        validate=completed_bars_only,
    )
    assert result.rows_published == 1 and result.rows_quarantined == 1
    assert result.quarantined[0].extra["reasons"] == [store_mod.QUARANTINE_VALIDATOR]


def test_health_surfaces_the_quarantine_count(store):
    store.publish("bar_m5", [_bar("AAPL", 0), _bar("NVDA", 0, close="bad")])
    counts = store.health_counts()
    assert counts["quarantine_files"] == 1 and counts["quarantine_rows"] == 1
    assert counts["live_rows"] == 1
    # Quarantine files are evidence, not dataset content.
    assert store.read_table("bar_m5").num_rows == 1
    assert counts["unmanifested_live_files"] == 0


def test_an_all_dirty_publish_writes_no_dataset_file(store):
    result = store.publish("bar_m5", [_bar("AAPL", 0, close="bad")])
    assert result.published == [] and result.rows_quarantined == 1
    assert store.manifest.resolve("bar_m5").entries == ()
    assert store.read_table("bar_m5").num_rows == 0


def test_repeated_dirty_publishes_keep_every_quarantined_record(store):
    for _ in range(3):
        store.publish("bar_m5", [_bar("NVDA", 0, close="bad")])
    entries = store.manifest.quarantine_entries()
    assert len(entries) == 3
    assert all((store.root / entry.file_path).exists() for entry in entries)
    assert store.health_counts()["quarantine_rows"] == 3
