"""Retire-and-rebuild of a month's derived partitions (BD-96, second half).

``bar_derived`` and ``feature_snapshot_intraday`` for August and September 2026
were computed from a ``bar_m5`` that was 85% duplicate rows. They carry no grain
duplicates of their own - the aggregator counted every twin as a constituent -
so a dedupe cannot repair them; only a recomputation from the repaired M5 rows
can. That recomputation must retire the old partition first (the feature step
skips ``ALREADY_COMPUTED`` keys), and retiring must be a manifest line, never a
delete.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from scripts.research_warehouse import cli
from scripts.research_warehouse.schemas import SCHEMA_VERSION
from scripts.research_warehouse.store import ResearchStore

UTC = timezone.utc
# 2026-08-03 is a Monday; 13:30 UTC is the 09:30 ET open.
OPEN = datetime(2026, 8, 3, 13, 30, tzinfo=UTC)


def _bar(minute: int, symbol: str = "AAPL") -> dict:
    start = OPEN + timedelta(minutes=minute)
    return {
        "symbol": symbol,
        "interval_start": start,
        "interval_end": start + timedelta(minutes=5),
        "session_id": "XNYS-2026-08-03",
        "session_phase": "RTH",
        "open": 100.0,
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
        "observed_at": start + timedelta(minutes=6),
        "capture_mode": "LIVE",
        "revision_id": "",
        "supersedes_revision_id": "",
        "schema_version": SCHEMA_VERSION,
        "run_id": "cycle-1",
    }


@pytest.fixture()
def lake(tmp_path):
    store = ResearchStore.open(tmp_path / "lake")
    # Three completed M5 bars, each published TWICE - the 2026-09-03 defect.
    rows = [_bar(0), _bar(5), _bar(10)]
    store.publish("bar_m5", rows)
    store.publish("bar_m5", rows)
    return store


def test_retire_partition_is_one_manifest_line_and_no_delete(lake):
    store = lake
    before = store.read_table("bar_m5").num_rows
    assert before == 6
    files = store.retire_partition("bar_m5", "month=2026-08", job_id="t", reason="test")
    assert len(files) == 2
    assert store.read_table("bar_m5").num_rows == 0, "retired files leave the live set at once"
    assert set(store.retired_pending()) == set(files), "...but stay on disk until GC"
    assert all((store.root / path).exists() for path in files)
    last = store.manifest.read_entries()[-1]
    assert last.action == "RETIRE" and list(last.supersedes) == files and last.extra.get("reason") == "test"
    # Nothing left to retire: no second line.
    ledger = store.manifest.path.read_bytes()
    assert store.retire_partition("bar_m5", "month=2026-08") == []
    assert store.manifest.path.read_bytes() == ledger


def test_retire_partition_refuses_bronze_evidence(lake, monkeypatch):
    from scripts.research_warehouse import schemas

    spec = schemas.DATASETS["bar_m5"]
    frozen = schemas.DatasetSpec(
        name=spec.name,
        layer=spec.layer,
        schema=spec.schema,
        time_column=spec.time_column,
        partition_by=spec.partition_by,
        grain=spec.grain,
        compactable=False,
    )
    monkeypatch.setitem(schemas.DATASETS, "bar_m5", frozen)
    with pytest.raises(ValueError, match="never retired wholesale"):
        lake.retire_partition("bar_m5", "month=2026-08")


def test_rebuild_month_dry_run_plans_and_writes_nothing(lake):
    store = lake
    now = datetime(2026, 8, 4, 22, 0, tzinfo=UTC)
    # Build the derived rows once, the way the nightly does, over the duplicated M5.
    from scripts.research_warehouse.aggregate import build_derived_bars

    build_derived_bars(store, [OPEN.date()], as_of=now, now=now, run_id="nightly")
    polluted = store.read_rows("bar_derived", "timeframe=M15/month=2026-08")
    assert polluted and polluted[0]["constituent_count"] == 6, "the defect: every twin counted"

    ledger = store.manifest.path.read_bytes()
    plan = cli.run_rebuild_month(store, month="2026-08", now=now)
    assert plan["applied"] is False
    assert plan["partitions"] == ["bar_derived/timeframe=M15/month=2026-08"] or "bar_derived/timeframe=M15/month=2026-08" in plan["partitions"]
    assert "2026-08-03" in plan["sessions"] and "2026-08-01" not in plan["sessions"], "exchange sessions only"
    assert store.manifest.path.read_bytes() == ledger


def test_rebuild_month_recomputes_from_the_repaired_bars(lake):
    store = lake
    now = datetime(2026, 8, 4, 22, 0, tzinfo=UTC)
    from scripts.research_warehouse.aggregate import build_derived_bars

    build_derived_bars(store, [OPEN.date()], as_of=now, now=now, run_id="nightly")
    assert store.read_rows("bar_derived", "timeframe=M15/month=2026-08")[0]["constituent_count"] == 6

    # Repair the source first (what the runbook says), then rebuild the month.
    assert store.dedupe_partition("bar_m5", "month=2026-08", job_id="dedupe").rows_dropped == 3
    report = cli.run_rebuild_month(store, month="2026-08", apply=True, now=now)

    assert report["status"] == "OK" and report["retired_files"] >= 1
    rebuilt = store.read_rows("bar_derived", "timeframe=M15/month=2026-08")
    assert len(rebuilt) == 1 and rebuilt[0]["constituent_count"] == 3 and rebuilt[0]["volume"] == 3000
    assert rebuilt[0]["run_id"] == "rebuild_month"
    # The polluted part file is retired, not deleted, and the ledger says why.
    retire_lines = [e for e in store.manifest.read_entries() if e.action == "RETIRE"]
    assert retire_lines and "rebuild 2026-08" in str(retire_lines[0].extra.get("reason"))


def test_rebuild_month_is_inert_without_a_store_and_refuses_a_bad_month():
    assert cli.run_rebuild_month(None, month="2026-08")["status"] == "DISABLED"
    store = None
    assert cli.run_rebuild_month(store, month="2026-13")["status"] == "DISABLED"
