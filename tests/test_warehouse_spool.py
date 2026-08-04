"""The machine-local capture spool (plan sec 8.4, 19.3 checklist).

Pinned here: the writer/sealer ownership split (rollover contract), the
5 GB / 7-day cap, the fixed shedding order, that D1/M5 capture is never shed,
and that a shed segment leaves an explicit gap record rather than a silent
hole. The spool is what keeps a DAS outage to "≤1 session of raw capture",
so its failure modes are tested, not assumed.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.research_warehouse import spool as spool_mod
from scripts.research_warehouse.schemas import SCHEMA_VERSION
from scripts.research_warehouse.spool import ResearchSpoolWriter, seal_spool
from scripts.research_warehouse.store import ResearchStore

UTC = timezone.utc
NOW = datetime(2026, 8, 3, 17, 0, tzinfo=UTC)


def _bar_row(symbol="AAPL", minute=0):
    start = datetime(2026, 8, 3, 13, 30, tzinfo=UTC) + timedelta(minutes=minute)
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
def writer(tmp_path):
    return ResearchSpoolWriter(tmp_path / "research_spool")


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


def test_writer_and_sealer_never_touch_the_same_file(writer, store):
    writer.write("bar_m5", [_bar_row(minute=0)], now=NOW)
    # While the segment is open the sealer sees nothing at all.
    assert spool_mod.closed_segments(writer.dir) == []
    assert seal_spool(store, writer.dir).status == "NOTHING_TO_SEAL"
    assert store.read_table("bar_m5").num_rows == 0

    closed = writer.roll()
    assert closed is not None and closed.name.endswith(spool_mod.CLOSED_SUFFIX)
    result = seal_spool(store, writer.dir)
    assert result.segments_sealed == 1 and result.rows_published == 1
    assert store.read_table("bar_m5").num_rows == 1
    assert list(writer.dir.glob("segment-*.jsonl")) == []


def test_rollover_on_size_and_on_age(tmp_path):
    writer = ResearchSpoolWriter(tmp_path / "spool", segment_max_bytes=200, segment_max_age_seconds=60)
    writer.write("bar_m5", [_bar_row(minute=index) for index in range(4)], now=NOW)
    # The next write finds the segment over size and starts a new one.
    writer.write("bar_m5", [_bar_row(minute=9)], now=NOW)
    assert len(spool_mod.closed_segments(writer.dir)) == 1

    writer.write("bar_m5", [_bar_row(minute=10)], now=NOW + timedelta(seconds=120))
    assert len(spool_mod.closed_segments(writer.dir)) >= 2
    assert len(list(writer.dir.glob(f"*{spool_mod.OPEN_SUFFIX}"))) == 1


def test_a_crashed_writer_leaves_its_segment_for_the_next_writer(tmp_path):
    first = ResearchSpoolWriter(tmp_path / "spool")
    first.write("bar_m5", [_bar_row()], now=NOW)
    open_segments = list(first.dir.glob(f"*{spool_mod.OPEN_SUFFIX}"))
    assert len(open_segments) == 1  # the process dies here

    second = ResearchSpoolWriter(tmp_path / "spool")
    assert list(second.dir.glob(f"*{spool_mod.OPEN_SUFFIX}")) == []
    assert len(spool_mod.closed_segments(second.dir)) == 1


def test_sealed_rows_round_trip_through_json(writer, store):
    writer.write("bar_m5", [_bar_row(minute=0), _bar_row(minute=5)], now=NOW)
    writer.roll()
    seal_spool(store, writer.dir)

    rows = store.read_table("bar_m5").to_pylist()
    assert len(rows) == 2
    assert rows[0]["interval_start"] == datetime(2026, 8, 3, 13, 30, tzinfo=UTC)
    assert rows[0]["provider"] == "IBKR" and rows[0]["is_complete"] is True


def test_naive_timestamps_are_refused_at_the_spool_boundary(writer):
    row = _bar_row()
    row["interval_start"] = datetime(2026, 8, 3, 13, 30)  # no timezone
    with pytest.raises(ValueError, match="timezone-aware"):
        writer.write("bar_m5", [row], now=NOW)


def test_shedding_order_is_fixed_and_d1_m5_never_shed(tmp_path):
    writer = ResearchSpoolWriter(tmp_path / "spool")

    writer.write("bar_m5", [_bar_row(minute=0)], now=NOW, shed_class=spool_mod.SHED_PROTECTED)
    writer.roll()
    writer.write("bar_m1", [_bar_row(minute=1)], now=NOW, shed_class=spool_mod.SHED_ETH_BARS)
    writer.roll()
    writer.write("bar_m1", [_bar_row(minute=2)], now=NOW, shed_class=spool_mod.SHED_M1_NON_FOCUS)
    writer.roll()
    writer.write("bar_m1", [_bar_row(minute=3)], now=NOW, shed_class=spool_mod.SHED_M1_EXPLORATION)
    writer.roll()

    writer.cap_bytes = 1  # the DAS is gone and the spool is over its cap
    shed = writer.enforce_cap(now=NOW)
    classes = [record["shed_class"] for record in shed]
    # Exploration extras first, then non-Focus M1, then ETH bars...
    assert classes == [
        spool_mod.SHED_M1_EXPLORATION,
        spool_mod.SHED_M1_NON_FOCUS,
        spool_mod.SHED_ETH_BARS,
    ]
    # ...and the protected M5 segment survives even with the cap at 1 byte.
    remaining = spool_mod.closed_segments(writer.dir)
    assert len(remaining) == 1
    assert json.loads(remaining[0].read_text(encoding="utf-8").splitlines()[0])["dataset"] == "bar_m5"


def test_shed_evidence_becomes_an_explicit_gap_row(tmp_path, store):
    writer = ResearchSpoolWriter(tmp_path / "spool", cap_bytes=1)
    writer.write("bar_m1", [_bar_row("NVDA", minute=1)], now=NOW, shed_class=spool_mod.SHED_M1_EXPLORATION)
    writer.roll()
    writer.enforce_cap(now=NOW)
    assert (writer.dir / spool_mod.SHED_LOG_NAME).exists()

    result = seal_spool(store, writer.dir)
    assert result.gaps_recorded == 1
    gap = store.read_table("collection_gap").to_pylist()[0]
    assert gap["symbol"] == "NVDA"
    # Shed-by-policy is policy absence, never MISSING.
    assert gap["reason"] == "NOT_COLLECTED_BY_POLICY" and gap["resolution"] == "POLICY"
    assert not (writer.dir / spool_mod.SHED_LOG_NAME).exists()


def test_age_cap_sheds_segments_older_than_seven_days(tmp_path):
    import os

    writer = ResearchSpoolWriter(tmp_path / "spool")
    writer.write("bar_m1", [_bar_row()], now=NOW, shed_class=spool_mod.SHED_M1_EXPLORATION)
    stale = writer.roll()
    old = (NOW - timedelta(days=8)).timestamp()
    os.utime(stale, (old, old))

    shed = writer.enforce_cap(now=NOW)
    assert [record["reason"] for record in shed] == ["AGE_CAP"]
    assert spool_mod.closed_segments(writer.dir) == []


def test_a_mixed_segment_is_treated_as_protected(tmp_path):
    writer = ResearchSpoolWriter(tmp_path / "spool", cap_bytes=1)
    writer.write("bar_m1", [_bar_row()], now=NOW, shed_class=spool_mod.SHED_M1_EXPLORATION)
    writer.write("bar_m5", [_bar_row()], now=NOW, shed_class=spool_mod.SHED_PROTECTED)
    writer.roll()

    assert writer.enforce_cap(now=NOW) == []
    assert len(spool_mod.closed_segments(writer.dir)) == 1


def test_stats_feed_the_health_tile(writer):
    writer.write("bar_m5", [_bar_row()], now=NOW)
    stats = writer.stats(now=NOW + timedelta(seconds=30))
    assert stats.segments == 1 and stats.open_segments == 1 and stats.closed_segments == 0
    assert stats.bytes > 0 and stats.shed_records == 0
    writer.roll()
    assert writer.stats(now=NOW).closed_segments == 1


def test_seal_is_a_no_op_when_the_warehouse_is_disabled(writer):
    writer.write("bar_m5", [_bar_row()], now=NOW)
    writer.roll()
    result = seal_spool(None, writer.dir)
    assert result.status == "DISABLED" and result.rows_published == 0
    assert len(spool_mod.closed_segments(writer.dir)) == 1  # nothing lost


def test_a_failed_publish_keeps_the_segment_for_the_next_run(writer, store, monkeypatch):
    writer.write("bar_m5", [_bar_row()], now=NOW)
    writer.roll()

    def explode(*args, **kwargs):
        raise OSError("DAS went away mid-seal")

    monkeypatch.setattr(store, "publish", explode)
    result = seal_spool(store, writer.dir)
    assert result.segments_sealed == 0 and result.segments_failed
    assert len(spool_mod.closed_segments(writer.dir)) == 1

    monkeypatch.undo()
    retry = seal_spool(store, writer.dir)
    assert retry.segments_sealed == 1 and store.read_table("bar_m5").num_rows == 1


def test_torn_segment_tail_seals_the_complete_records(writer, store):
    writer.write("bar_m5", [_bar_row(minute=0)], now=NOW)
    segment = writer.roll()
    with open(segment, "a", encoding="utf-8") as handle:
        handle.write('{"dataset": "bar_m5", "row": {"symbol": "AA')  # killed mid-write

    result = seal_spool(store, writer.dir)
    assert result.rows_published == 1
    assert store.read_table("bar_m5").num_rows == 1
