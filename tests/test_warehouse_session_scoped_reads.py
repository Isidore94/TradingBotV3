"""The warehouse build must not materialise a whole month to use one session.

Measured on the desk 2026-08-27: the desk process climbed to 8-13 GB after
every hourly swing-scan slot and fell back minutes later. Three steps of the
post-scan build did `store.read_table("bar_m5", "month=YYYY-MM").to_pylist()`
and then kept ONE session, and the month partition is month-keyed so it grows
all month - `silver/bar_m5/month=2026-08` reached **8,704,108 rows / 408 MB
parquet**, and `to_pylist` costs **1,769 bytes per row = 15.4 GB** if the month
is held whole. The largest single session in that month is 588,778 rows, or
6.8% of it.

So these tests are about SIZE, not just output: each one asserts that the rows
crossing into Python scale with the slice the step actually needs. The
equivalence tests beside them assert the published rows are unchanged, because
a cheaper read that quietly drops a row would be a far worse bug than the one
being fixed.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("pyarrow", reason="the research lake is parquet")


# --------------------------------------------------------------------- lake


@pytest.fixture
def lake(tmp_path, monkeypatch):
    """A real ResearchStore on a temp root - no mocks of the read path.

    The bug lives in how much a real Arrow read hands to Python, so a fake
    store would test the wrong thing.
    """
    from research_warehouse import config
    from research_warehouse.store import ResearchStore

    root = tmp_path / "research_lake"
    monkeypatch.setattr(config, "get_research_store_dir", lambda: root)
    monkeypatch.setattr(config, "warehouse_enabled", lambda: True)
    config.ensure_lake_layout(root)
    return ResearchStore(root)


def _m5_rows(session, symbols, *, count=12, run_id="fixture"):
    """`count` five-minute bars per symbol, starting at the RTH open."""
    rows = []
    for symbol in symbols:
        price = 100.0 + len(symbol)
        for index in range(count):
            start = session.rth_open_at + timedelta(minutes=5 * index)
            rows.append(
                {
                    "symbol": symbol,
                    "interval_start": start,
                    "interval_end": start + timedelta(minutes=5),
                    "session_id": session.session_id,
                    "session_phase": "RTH",
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price + 0.1 * index,
                    "volume": 1000 + index,
                    "vwap": price,
                    "trade_count": 10,
                    "provider": "fixture",
                    "is_complete": True,
                    "quality": "COMPLETE",
                    "source_hash": f"{symbol}-{index}",
                    "event_at": start,
                    "observed_at": start,
                    "capture_mode": "BACKFILL",
                    "revision_id": f"{symbol}-{index}",
                    "supersedes_revision_id": "",
                    "schema_version": "1",
                    "run_id": run_id,
                }
            )
    return rows


def _two_sessions():
    """Two consecutive trading sessions in one month partition."""
    from research_warehouse import exchange_calendar as xcal

    found = []
    day = date(2026, 8, 3)
    while len(found) < 2 and day < date(2026, 8, 28):
        session = xcal.trading_session(day)
        if session is not None:
            found.append((day, session))
        day += timedelta(days=1)
    assert len(found) == 2, "the fixture needs two trading days in one month"
    assert found[0][1].rth_open_at.strftime("%Y-%m") == found[1][1].rth_open_at.strftime("%Y-%m")
    return found


# ------------------------------------------------- the store-level helper


def test_read_rows_filters_in_arrow_before_python_sees_it(lake):
    """The helper is the whole fix: the predicate runs in Arrow, so the rows
    that become Python dicts are only the ones asked for."""
    (day_a, session_a), (day_b, session_b) = _two_sessions()
    partition = f"month={session_a.rth_open_at:%Y-%m}"
    lake.publish(
        "bar_m5",
        _m5_rows(session_a, ["AAA", "BBB"]) + _m5_rows(session_b, ["AAA", "BBB"]),
        job_id="fixture",
    )

    everything = lake.read_rows("bar_m5", partition)
    assert len(everything) == 48, "no filter reads the partition, as read_table did"

    one_session = lake.read_rows(
        "bar_m5",
        partition,
        interval_start_range=(session_a.rth_open_at, session_a.rth_close_at),
    )
    assert len(one_session) == 24
    assert {row["session_id"] for row in one_session} == {session_a.session_id}

    one_symbol = lake.read_rows(
        "bar_m5",
        partition,
        interval_start_range=(session_a.rth_open_at, session_a.rth_close_at),
        symbols=["AAA"],
    )
    assert len(one_symbol) == 12
    assert {row["symbol"] for row in one_symbol} == {"AAA"}


def test_read_rows_matches_read_table_row_for_row(lake):
    """Cheaper must mean IDENTICAL, not merely similar."""
    (day_a, session_a), (_day_b, session_b) = _two_sessions()
    partition = f"month={session_a.rth_open_at:%Y-%m}"
    lake.publish(
        "bar_m5",
        _m5_rows(session_a, ["AAA", "BBB"]) + _m5_rows(session_b, ["AAA"]),
        job_id="fixture",
    )

    old_way = [
        row
        for row in lake.read_table("bar_m5", partition).to_pylist()
        if row["symbol"] in {"AAA"}
        and session_a.rth_open_at <= row["interval_start"] < session_a.rth_close_at
    ]
    new_way = lake.read_rows(
        "bar_m5",
        partition,
        interval_start_range=(session_a.rth_open_at, session_a.rth_close_at),
        symbols=["AAA"],
    )
    assert new_way == old_way
    assert new_way, "the fixture must actually produce rows"


def test_read_rows_on_an_empty_partition_is_an_empty_list(lake):
    """A partition with no manifest-live files must not raise - `read_table`
    returns an empty table there and every caller relies on it."""
    assert lake.read_rows("bar_m5", "month=1999-01") == []
    assert lake.read_rows("bar_m5", "month=1999-01", symbols=["AAA"]) == []


def test_read_rows_with_an_empty_symbol_list_means_no_symbol_filter(lake):
    """`symbols=[]` must behave like `symbols=None`, because that is exactly
    what the callers pass when the trader asked for the whole cohort."""
    (_day_a, session_a), _b = _two_sessions()
    partition = f"month={session_a.rth_open_at:%Y-%m}"
    lake.publish("bar_m5", _m5_rows(session_a, ["AAA", "BBB"]), job_id="fixture")

    assert len(lake.read_rows("bar_m5", partition, symbols=[])) == 24
    assert len(lake.read_rows("bar_m5", partition, symbols=None)) == 24


def test_read_rows_can_still_select_columns(lake):
    (_day_a, session_a), _b = _two_sessions()
    partition = f"month={session_a.rth_open_at:%Y-%m}"
    lake.publish("bar_m5", _m5_rows(session_a, ["AAA"]), job_id="fixture")

    rows = lake.read_rows("bar_m5", partition, columns=["symbol", "interval_start"])
    assert rows and set(rows[0]) == {"symbol", "interval_start"}


# ------------------------------------------------------- the three readers


def _rows_handed_to_python(monkeypatch, store):
    """Count every row that crosses the Arrow -> Python boundary in a build.

    This is the actual defect: not the answer, but how much had to be
    materialised to reach it.
    """
    seen: list[int] = []
    real_read_rows = type(store).read_rows
    real_read_table = type(store).read_table

    def counting_read_rows(self, dataset, partition=None, **kwargs):
        rows = real_read_rows(self, dataset, partition, **kwargs)
        if dataset in ("bar_m5", "bar_derived"):
            seen.append(len(rows))
        return rows

    class _CountingTable:
        def __init__(self, table, dataset):
            self._table = table
            self._dataset = dataset

        def to_pylist(self):
            rows = self._table.to_pylist()
            if self._dataset in ("bar_m5", "bar_derived"):
                seen.append(len(rows))
            return rows

        def __getattr__(self, name):
            return getattr(self._table, name)

    def counting_read_table(self, dataset, partition=None, columns=None):
        return _CountingTable(real_read_table(self, dataset, partition, columns), dataset)

    monkeypatch.setattr(type(store), "read_rows", counting_read_rows, raising=False)
    monkeypatch.setattr(type(store), "read_table", counting_read_table)
    return seen


def test_build_derived_bars_materialises_one_session_not_the_month(lake, monkeypatch):
    from research_warehouse import aggregate

    (day_a, session_a), (day_b, session_b) = _two_sessions()
    lake.publish(
        "bar_m5",
        _m5_rows(session_a, ["AAA", "BBB"]) + _m5_rows(session_b, ["AAA", "BBB"]),
        job_id="fixture",
    )

    seen = _rows_handed_to_python(monkeypatch, lake)
    report = aggregate.build_derived_bars(lake, [day_a], symbols=["AAA"])

    assert report.rows_published > 0, "the fixture must actually derive something"
    assert max(seen) <= 12, (
        f"one session of one symbol is 12 bars; Python saw {max(seen)} rows "
        "- the whole month was materialised again"
    )


def test_build_derived_bars_publishes_exactly_what_the_old_read_produced(lake, monkeypatch):
    """Equivalence against a REFERENCE implementation of the old read.

    Compared as published ROWS, not counts: a filter that shifted a session
    boundary by one bar would keep the count and change the answer. The
    reference below is the pre-fix code path written out longhand - read the
    whole partition, filter in Python - so this stays a real comparison after
    the production code stops doing that.
    """
    from research_warehouse import aggregate
    from research_warehouse.aggregate import derive_session_bars

    (day_a, session_a), (_day_b, session_b) = _two_sessions()
    lake.publish(
        "bar_m5",
        _m5_rows(session_a, ["AAA", "BBB"]) + _m5_rows(session_b, ["AAA", "BBB"]),
        job_id="fixture",
    )
    partition = f"month={session_a.rth_open_at:%Y-%m}"

    # --- the old way, longhand -----------------------------------------
    wanted = {"AAA"}
    by_symbol: dict[str, list[dict]] = {}
    for row in lake.read_table("bar_m5", partition).to_pylist():
        symbol = str(row.get("symbol") or "")
        if wanted and symbol not in wanted:
            continue
        start = row.get("interval_start")
        if start is None:
            continue
        stamp = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        if session_a.rth_open_at <= stamp < session_a.rth_close_at:
            by_symbol.setdefault(symbol, []).append(row)
    stamp_now = datetime(2026, 8, 27, 20, 0, tzinfo=timezone.utc)
    expected = []
    for timeframe in aggregate.SLICE_TIMEFRAMES:
        for symbol, symbol_rows in sorted(by_symbol.items()):
            expected.extend(
                derive_session_bars(
                    symbol_rows, session_a, timeframe,
                    as_of=stamp_now, computed_at=stamp_now, run_id="eq",
                )
            )
    assert expected, "the reference must actually derive something"

    # --- what the code does now -----------------------------------------
    published: list[dict] = []
    real_publish = type(lake).publish

    def capture(self, dataset, rows, **kwargs):
        if dataset == "bar_derived":
            published.extend(dict(row) for row in rows)
        return real_publish(self, dataset, rows, **kwargs)

    monkeypatch.setattr(type(lake), "publish", capture)
    aggregate.build_derived_bars(
        lake, [day_a], symbols=["AAA"], as_of=stamp_now, now=stamp_now, run_id="eq"
    )

    assert published == expected, "the cheaper read changed what gets published"


def test_intraday_snapshots_materialise_one_session_not_the_month(lake, monkeypatch):
    from research_warehouse import features

    (day_a, session_a), (_day_b, session_b) = _two_sessions()
    lake.publish(
        "bar_m5",
        _m5_rows(session_a, ["AAA", "BBB"]) + _m5_rows(session_b, ["AAA", "BBB"]),
        job_id="fixture",
    )

    seen = _rows_handed_to_python(monkeypatch, lake)
    features.build_intraday_snapshots(lake, day_a, symbols=["AAA"])

    assert seen, "the step must have read something"
    assert max(seen) <= 24, (
        f"one session is 24 bars across both symbols; Python saw {max(seen)} "
        "- the whole month was materialised again"
    )


def test_intraday_snapshots_still_see_every_symbol_when_none_are_named(lake, monkeypatch):
    """`symbols=None` means "the cohort present in this session", derived from
    the bars themselves - so the symbol filter must NOT be applied then."""
    from research_warehouse import features

    (day_a, session_a), (_day_b, session_b) = _two_sessions()
    lake.publish(
        "bar_m5",
        _m5_rows(session_a, ["AAA", "BBB"]) + _m5_rows(session_b, ["CCC"]),
        job_id="fixture",
    )

    published: list[list[dict]] = []
    real_publish = type(lake).publish

    def capture(self, dataset, rows, **kwargs):
        published.append([dict(row) for row in rows])
        return real_publish(self, dataset, rows, **kwargs)

    monkeypatch.setattr(type(lake), "publish", capture)
    features.build_intraday_snapshots(lake, day_a)

    assert published, "nothing published"
    symbols = {row["symbol"] for row in published[0]}
    assert symbols == {"AAA", "BBB"}, (
        "both symbols of that session must be snapshotted, and CCC (a different "
        f"session) must not - got {symbols}"
    )


def test_outcomes_reads_only_the_occurrence_symbols(lake, monkeypatch):
    """`_run_outcomes` filters by symbol and NOT by day - the outcome walk runs
    forward over a horizon that crosses sessions. So the symbol filter is the
    only one that can move into Arrow without changing the answer."""
    from research_warehouse import cli

    (day_a, session_a), (_day_b, session_b) = _two_sessions()
    lake.publish(
        "bar_m5",
        _m5_rows(session_a, ["AAA", "ZZZ"]) + _m5_rows(session_b, ["AAA", "ZZZ"]),
        job_id="fixture",
    )

    known = {
        ("AAA", "x"): {
            "symbol": "AAA",
            "trigger_at": session_a.rth_open_at + timedelta(minutes=10),
        }
    }
    monkeypatch.setattr(
        cli.occurrences, "latest_occurrences", lambda store, year: dict(known) if year == day_a.year else {}
    )
    captured: dict = {}

    def fake_build_outcomes(store, occ, **kwargs):
        captured["m5"] = kwargs["m5_by_symbol"]
        from research_warehouse.outcomes import OutcomeReport

        return OutcomeReport()

    monkeypatch.setattr(cli.outcomes, "build_outcomes", fake_build_outcomes)
    monkeypatch.setattr(cli.features, "daily_history_window", lambda store, day: ([], {}))
    monkeypatch.setattr(cli, "_bands_by_occurrence", lambda store, known: {})

    seen = _rows_handed_to_python(monkeypatch, lake)
    cli._run_outcomes(lake, day_a, datetime.now(timezone.utc), "run")

    assert set(captured["m5"]) == {"AAA"}, "only occurrence symbols reach the walk"
    assert sum(len(v) for v in captured["m5"].values()) == 24, (
        "BOTH sessions of AAA must survive - the walk crosses sessions"
    )
    assert max(seen) <= 24, (
        f"only AAA's rows should cross into Python; saw {max(seen)} "
        "(ZZZ's rows were materialised too)"
    )
