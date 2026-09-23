"""Benchmark D1 bars in the lake, and Auto Market Bias context that reads them.

Pinned here: the nightly bar_d1 ingest always carries SPY/QQQ/IWM (they are
never universe members, so silver bar_d1 had no SPY and every D1/M5 market
context row was unknown); the two repair commands are dry runs by default,
write nothing then, are idempotent when applied, and use only bars completed
by each occurrence's entry; readers prefer the newest bias definition.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from research_warehouse import cli, market_bias_context, schemas  # noqa: E402
from research_warehouse import exchange_calendar as xcal  # noqa: E402
from research_warehouse import ingest_existing as ingest  # noqa: E402
from research_warehouse.store import ResearchStore  # noqa: E402

UTC = timezone.utc
FIRST_DAY = date(2026, 6, 22)
HISTORY_SESSIONS = 30
M5_SESSIONS = 14


def _sessions(count: int, start: date = FIRST_DAY) -> list:
    out, day = [], start
    while len(out) < count:
        session = xcal.trading_session(day)
        if session is not None:
            out.append(session)
        day += timedelta(days=1)
    return out


SESSIONS = _sessions(HISTORY_SESSIONS + 3)
HISTORY = SESSIONS[:HISTORY_SESSIONS]
ENTRY_SESSION = SESSIONS[HISTORY_SESSIONS]
LATER_SESSIONS = SESSIONS[HISTORY_SESSIONS + 1:]
TODAY = LATER_SESSIONS[-1].session_date + timedelta(days=1)
NOW = datetime(TODAY.year, TODAY.month, TODAY.day, 2, 0, tzinfo=UTC)


def _d1_close(index: int) -> float:
    return 500.0 + index


def _write_durable(bars_dir: Path, symbol: str = "SPY", *, future: bool = True) -> None:
    import pandas as pd

    sessions = HISTORY + ([ENTRY_SESSION, *LATER_SESSIONS] if future else [])
    records = []
    for index, session in enumerate(sessions):
        close = _d1_close(index) if index < HISTORY_SESSIONS else 9_999.0  # future: absurd
        records.append(
            {
                "datetime": datetime.combine(session.session_date, datetime.min.time()),
                "open": close - 1, "high": close + 2, "low": close - 2, "close": close,
                "volume": 10_000_000,
            }
        )
    bars_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(ingest.durable_daily_bar_file(symbol, bars_dir), index=False)


def _m5_rows(session, *, base: float, count: int = 78, step: float = 0.01, symbol: str = "SPY") -> list[dict]:
    rows = []
    for index in range(count):
        start = session.rth_open_at + timedelta(minutes=5 * index)
        close = base + index * step
        rows.append(
            {
                "symbol": symbol,
                "interval_start": start,
                "interval_end": start + timedelta(minutes=5),
                "session_id": session.session_id,
                "session_phase": "RTH",
                "open": close, "high": close + 0.2, "low": close - 0.2, "close": close,
                "volume": 1000, "vwap": None, "trade_count": None,
                "provider": "IBKR", "is_complete": True, "quality": "COMPLETE", "source_hash": "",
                "event_at": start + timedelta(minutes=5), "observed_at": start + timedelta(minutes=6),
                "capture_mode": "BACKFILL", "revision_id": "", "supersedes_revision_id": "",
                "schema_version": schemas.SCHEMA_VERSION, "run_id": "test",
            }
        )
    return rows


def _occurrence(occurrence_id: str = "occ-1", symbol: str = "AAA") -> dict:
    # Known after the last history session closed; entry is the next session's first bar.
    trigger = HISTORY[-1].rth_close_at + timedelta(hours=1)
    return {
        "occurrence_id": occurrence_id,
        "symbol": symbol,
        "canonical_setup_id": "AVWAP_BREAKOUT",
        "side": "LONG",
        "structural_timeframe": "D1",
        "trigger_timeframe": "D1",
        "anchor_instance_id": "",
        "dependency_cluster_id": occurrence_id,
        "status": "OPEN",
        "trigger_at": trigger,
        "trigger_bar_interval_start": trigger,
        "entry_price_ref": 100.0,
        "stop_price_ref": 95.0,
        "detector_version": "test",
        "first_detected_run_id": "run-1",
        "last_updated_run_id": "run-1",
        "tags": "",
        "event_at": trigger,
        "observed_at": trigger,
        "computed_at": trigger,
        "revision_id": f"rev-{occurrence_id}",
        "supersedes_revision_id": "",
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "run-1",
    }


def _lake_files(store: ResearchStore) -> dict[str, int]:
    return {
        str(path.relative_to(store.root)): path.stat().st_size
        for path in store.root.rglob("*")
        if path.is_file()
    }


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


@pytest.fixture()
def bars_dir(tmp_path):
    target = tmp_path / "daily_bars"
    for symbol in ("SPY", "QQQ", "IWM"):
        _write_durable(target, symbol)
    return target


def _seed_context_lake(store: ResearchStore, *, future_m5: bool = True) -> None:
    m5 = []
    for index, session in enumerate(HISTORY[-M5_SESSIONS:]):
        m5.extend(_m5_rows(session, base=520.0 + index))
    m5.extend(_m5_rows(ENTRY_SESSION, base=540.0, count=3))
    if future_m5:
        m5.extend(_m5_rows(ENTRY_SESSION, base=9_000.0, count=78)[3:])
        for session in LATER_SESSIONS:
            m5.extend(_m5_rows(session, base=9_000.0))
    store.publish("bar_m5", m5, job_id="test")
    store.publish("setup_occurrence", [_occurrence()], job_id="test")


# --- 1. the nightly cohort ------------------------------------------------
def test_benchmarks_are_one_named_constant():
    assert ingest.BENCHMARK_SYMBOLS == ("SPY", "QQQ", "IWM")


def test_the_nightly_d1_cohort_always_carries_the_benchmarks(store):
    # No universe membership at all: SPY is never a member, yet it must be ingested.
    cohort = cli.d1_ingest_cohort(store, date(2026, 8, 3))
    assert set(ingest.BENCHMARK_SYMBOLS) <= set(cohort)


def test_the_build_publishes_spy_d1_completed_bars_only(store, bars_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest._paths(), "MASTER_AVWAP_DAILY_BARS_DIR", bars_dir, raising=False)
    day = ENTRY_SESSION.session_date
    report = cli.run_build(store, session_date=day, now=NOW, lock_path=tmp_path / "lock")
    assert report.status == "OK"
    rows = store.read_rows("bar_d1", symbols=["SPY"])
    assert len(rows) == HISTORY_SESSIONS
    assert max(row["session_date"] for row in rows) < day  # the forming bar is never evidence
    again = cli.run_build(store, session_date=day, now=NOW, lock_path=tmp_path / "lock")
    assert again.status == "OK"
    assert len(store.read_rows("bar_d1", symbols=["SPY"])) == HISTORY_SESSIONS


# --- 2. backfill-benchmark-d1 --------------------------------------------
def test_benchmark_backfill_dry_run_writes_nothing(store, bars_dir, tmp_path):
    before = _lake_files(store)
    lock = tmp_path / "lock"
    report = cli.run_backfill_benchmark_d1(store, bars_dir=bars_dir, now=NOW, lock_path=lock)
    assert report["applied"] is False
    assert report["status"] == "OK"
    total = HISTORY_SESSIONS + 1 + len(LATER_SESSIONS)  # every session before TODAY is complete
    assert report["rows"] == 3 * total
    spy = report["symbols"]["SPY"]
    assert spy["rows"] == total
    assert spy["first_session"] == HISTORY[0].session_date.isoformat()
    assert spy["last_session"] == LATER_SESSIONS[-1].session_date.isoformat()
    assert _lake_files(store) == before
    assert not lock.exists()


def test_benchmark_backfill_apply_is_idempotent_and_honest(store, bars_dir, tmp_path):
    first = cli.run_backfill_benchmark_d1(store, apply=True, bars_dir=bars_dir, now=NOW, lock_path=tmp_path / "lock")
    assert first["applied"] is True and first["rows"] > 0
    rows = store.read_rows("bar_d1", symbols=["SPY", "QQQ", "IWM"])
    assert len(rows) == first["rows"]
    assert {row["provider"] for row in rows} == {"UNKNOWN"}
    assert {row["capture_mode"] for row in rows} == {ingest.BRONZE_CAPTURE_MODE}
    assert {row["is_complete"] for row in rows} == {True}
    assert all(row["session_date"] < TODAY for row in rows)

    second = cli.run_backfill_benchmark_d1(store, apply=True, bars_dir=bars_dir, now=NOW, lock_path=tmp_path / "lock")
    assert second["rows"] == 0
    assert len(store.read_rows("bar_d1", symbols=["SPY", "QQQ", "IWM"])) == len(rows)


def test_benchmark_backfill_apply_refuses_while_a_build_runs(store, bars_dir, tmp_path):
    lock = tmp_path / "lock"
    with cli.single_flight(lock):
        report = cli.run_backfill_benchmark_d1(store, apply=True, bars_dir=bars_dir, now=NOW, lock_path=lock)
    assert report["status"] == "REFUSED"
    assert store.read_rows("bar_d1", symbols=["SPY"]) == []


# --- 3. market context v2 --------------------------------------------------
def test_the_bias_definition_is_versioned_and_v1_is_history():
    assert market_bias_context.BIAS_DEFINITION_ID == "auto_market_bias_multiframe_v2"
    assert market_bias_context.BIAS_DEFINITION_HISTORY[-1] == market_bias_context.BIAS_DEFINITION_ID
    assert "auto_market_bias_multiframe_v1" in market_bias_context.BIAS_DEFINITION_HISTORY


def _context(store) -> dict[str, dict]:
    return {
        row["timeframe"]: row
        for row in store.read_rows("setup_market_context")
        if row["bias_definition_id"] == market_bias_context.BIAS_DEFINITION_ID
    }


def test_v2_context_reads_spy_d1_and_m5_gets_the_previous_close(store, bars_dir, tmp_path):
    _seed_context_lake(store)
    cli.run_backfill_benchmark_d1(store, apply=True, bars_dir=bars_dir, now=NOW, lock_path=tmp_path / "lock")

    report = cli.run_backfill_market_context(store, apply=True, now=NOW, lock_path=tmp_path / "lock")
    assert report["status"] == "OK" and report["rows"] == 5

    context = _context(store)
    d1 = context["D1"]
    assert d1["env_key"] != "unknown"
    assert d1["bar_count"] == market_bias_context.ROLLING_BARS
    # The newest completed session before the entry session, never the entry day.
    assert d1["last_close"] == pytest.approx(_d1_close(HISTORY_SESSIONS - 1))
    m5 = context["M5"]
    assert m5["env_key"] != "unknown"
    assert m5["reference_close"] == pytest.approx(_d1_close(HISTORY_SESSIONS - 1))
    # Entry is the first completed RTH bar after the trigger: nothing later is read.
    assert m5["bar_count"] == 1
    assert m5["last_close"] == pytest.approx(540.0)
    assert m5["entry_at"] == ENTRY_SESSION.rth_open_at + timedelta(minutes=5)


def test_future_bars_change_nothing(store, bars_dir, tmp_path):
    _seed_context_lake(store, future_m5=True)
    cli.run_backfill_benchmark_d1(store, apply=True, bars_dir=bars_dir, now=NOW, lock_path=tmp_path / "lock")
    cli.run_backfill_market_context(store, apply=True, now=NOW, lock_path=tmp_path / "lock")
    with_future = _context(store)

    clean = ResearchStore.open(tmp_path / "clean_lake")
    clean_bars = tmp_path / "clean_bars"
    _write_durable(clean_bars, "SPY", future=False)
    _seed_context_lake(clean, future_m5=False)
    cli.run_backfill_benchmark_d1(clean, apply=True, bars_dir=clean_bars, now=NOW, lock_path=tmp_path / "lock")
    cli.run_backfill_market_context(clean, apply=True, now=NOW, lock_path=tmp_path / "lock")
    without_future = _context(clean)

    fields = ("env_key", "source", "last_close", "reference_close", "vwap", "bar_count")
    for timeframe in market_bias_context.TIMEFRAMES:
        assert {k: with_future[timeframe][k] for k in fields} == {
            k: without_future[timeframe][k] for k in fields
        }, timeframe


def test_context_backfill_dry_run_writes_nothing_and_apply_is_idempotent(store, bars_dir, tmp_path):
    _seed_context_lake(store)
    cli.run_backfill_benchmark_d1(store, apply=True, bars_dir=bars_dir, now=NOW, lock_path=tmp_path / "lock")
    before = _lake_files(store)
    lock = tmp_path / "lock"

    dry = cli.run_backfill_market_context(store, now=NOW, lock_path=lock)
    assert dry["applied"] is False and dry["rows"] == 5
    assert dry["unknown"].get("D1", 0) == 0
    assert _lake_files(store) == before
    assert not lock.exists()

    cli.run_backfill_market_context(store, apply=True, now=NOW, lock_path=lock)
    again = cli.run_backfill_market_context(store, apply=True, now=NOW, lock_path=lock)
    assert again["rows"] == 0
    assert len(store.read_rows("setup_market_context")) == 5


def test_old_unknown_v1_rows_do_not_block_v2_and_stay_as_history(store, bars_dir, tmp_path):
    _seed_context_lake(store)
    entry_at = ENTRY_SESSION.rth_open_at + timedelta(minutes=5)
    store.publish(
        "setup_market_context",
        [
            {
                "occurrence_id": "occ-1", "symbol": "AAA", "entry_at": entry_at, "timeframe": timeframe,
                "bias_definition_id": "auto_market_bias_multiframe_v1", "env_key": "unknown",
                "source": "insufficient_completed_bars", "bar_count": 0, "computed_at": entry_at,
                "schema_version": schemas.SCHEMA_VERSION, "run_id": "old",
            }
            for timeframe in market_bias_context.TIMEFRAMES
        ],
        job_id="test",
    )
    cli.run_backfill_benchmark_d1(store, apply=True, bars_dir=bars_dir, now=NOW, lock_path=tmp_path / "lock")
    report = cli.run_backfill_market_context(store, apply=True, now=NOW, lock_path=tmp_path / "lock")
    assert report["rows"] == 5
    rows = store.read_rows("setup_market_context")
    assert sum(row["bias_definition_id"] == "auto_market_bias_multiframe_v1" for row in rows) == 5
    newest = market_bias_context.newest_context_rows(rows)
    assert newest[("occ-1", "D1")]["bias_definition_id"] == market_bias_context.BIAS_DEFINITION_ID
    assert newest[("occ-1", "D1")]["env_key"] != "unknown"


def test_since_until_bound_the_walk(store, bars_dir, tmp_path):
    _seed_context_lake(store)
    cli.run_backfill_benchmark_d1(store, apply=True, bars_dir=bars_dir, now=NOW, lock_path=tmp_path / "lock")
    after = HISTORY[-1].session_date + timedelta(days=5)
    report = cli.run_backfill_market_context(store, since=after, now=NOW, lock_path=tmp_path / "lock")
    assert report["occurrences"] == 0 and report["rows"] == 0


def test_real_context_backfill_refuses_without_spy_d1_in_the_lake(store, bars_dir, tmp_path):
    _seed_context_lake(store)
    report = cli.run_backfill_market_context(store, apply=True, now=NOW, lock_path=tmp_path / "lock")
    assert report["status"] == "NO_SPY_D1"
    assert store.read_rows("setup_market_context") == []


def test_context_dry_run_can_preview_with_the_durable_d1_store(store, bars_dir, tmp_path):
    _seed_context_lake(store)
    before = _lake_files(store)
    report = cli.run_backfill_market_context(store, now=NOW, preview_bars_dir=bars_dir)
    assert report["spy_d1_source"] == "durable_store_preview"
    assert report["rows"] == 5 and report["unknown"].get("D1", 0) == 0
    assert _lake_files(store) == before


# --- readers ---------------------------------------------------------------
def test_newest_definition_wins_whatever_the_row_order():
    v1 = {"occurrence_id": "o", "timeframe": "D1", "bias_definition_id": "auto_market_bias_multiframe_v1", "env_key": "unknown"}
    v2 = {"occurrence_id": "o", "timeframe": "D1", "bias_definition_id": market_bias_context.BIAS_DEFINITION_ID, "env_key": "bullish_weak"}
    only_v1 = {"occurrence_id": "p", "timeframe": "D1", "bias_definition_id": "auto_market_bias_multiframe_v1", "env_key": "bearish_weak"}
    for rows in ([v1, v2, only_v1], [v2, v1, only_v1]):
        newest = market_bias_context.newest_context_rows(rows)
        assert newest[("o", "D1")]["env_key"] == "bullish_weak"
        assert newest[("p", "D1")]["env_key"] == "bearish_weak"


def test_setup_research_reads_the_newest_definition(store, monkeypatch):
    from ai_jobs import setup_research

    store.publish("setup_occurrence", [_occurrence(), _occurrence("occ-2")], job_id="test")
    entry_at = ENTRY_SESSION.rth_open_at + timedelta(minutes=5)
    base = {
        "occurrence_id": "occ-1", "symbol": "AAA", "entry_at": entry_at, "timeframe": "D1",
        "source": "x", "bar_count": 20, "computed_at": entry_at,
        "schema_version": schemas.SCHEMA_VERSION, "run_id": "t",
    }
    store.publish(
        "setup_market_context",
        [
            {**base, "bias_definition_id": "auto_market_bias_multiframe_v1", "env_key": "unknown"},
            {**base, "bias_definition_id": market_bias_context.BIAS_DEFINITION_ID, "env_key": "bullish_weak"},
            # Not yet recomputed: the older definition still answers.
            {**base, "occurrence_id": "occ-2", "timeframe": "H1",
             "bias_definition_id": "auto_market_bias_multiframe_v1", "env_key": "bearish_weak"},
        ],
        job_id="test",
    )
    monkeypatch.setattr(ResearchStore, "open", classmethod(lambda cls, root=None: store))
    monkeypatch.setattr(setup_research, "_now", lambda value=None: NOW)

    _latest, _occurrences, contexts, _coverage = setup_research._load()

    assert contexts["occ-1"]["D1"] == "bullish_weak"
    assert contexts["occ-2"]["H1"] == "bearish_weak"
