"""The chart bar cache and its off-thread data service (Part C, C3 + D4).

What these protect:

* the GUI thread never reads a file to draw a chart;
* a symbol is parsed from the shared store at most once per session, then
  served from memory, with a local mirror so even a cold start skips Drive;
* a fast sequence of symbol switches cannot paint an older chart last.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ui.services.bar_cache import BarSeries, D1BarStore


def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        return None
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _pump_until(predicate, timeout=10.0):
    app = _qt_app()
    if app is None:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


def _frame(rows=30, start=datetime(2026, 1, 1)):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "datetime": start + timedelta(days=index),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 1_000.0 + index,
            }
            for index in range(rows)
        ]
    )


def _series(symbol="TEST", rows=30):
    return BarSeries.from_frame(symbol, _frame(rows))


# ---------------------------------------------------------------- BarSeries
def test_series_round_trips_through_bar_dicts():
    series = _series(rows=5)
    bars = series.as_bar_dicts()
    assert len(bars) == 5
    assert bars[0]["open"] == 100.0
    assert bars[-1]["close"] == pytest.approx(104.5)
    # The dict shape must match what chart_snapshot's loader contract expects,
    # or the indicator math silently sees nothing.
    assert set(bars[0]) == {"dt", "open", "high", "low", "close", "volume"}
    assert isinstance(bars[0]["dt"], datetime)


def test_missing_volume_column_is_zeros_but_a_missing_price_raises():
    frame = _frame(4).drop(columns=["volume"])
    series = BarSeries.from_frame("NOVOL", frame)
    assert list(series.volume) == [0.0, 0.0, 0.0, 0.0]
    # A store missing a price column is a broken file; zeros would draw a lie.
    with pytest.raises(KeyError):
        BarSeries.from_frame("BROKEN", _frame(4).drop(columns=["close"]))


def test_live_bar_appends_and_replaces_the_forming_tail():
    series = _series(rows=3)
    nxt = {
        "dt": datetime(2026, 1, 4),
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": 1.5,
        "volume": 10.0,
    }
    grown = series.with_appended(nxt)
    assert len(grown) == 4 and grown.close[-1] == 1.5

    # The same stamp again is the SAME forming bar updating, not a new one.
    updated = grown.with_appended(dict(nxt, close=1.9))
    assert len(updated) == 4
    assert updated.close[-1] == 1.9


# ------------------------------------------------------------- D1BarStore
def test_shared_store_read_is_mirrored_then_served_from_memory(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    _frame(20).to_parquet(shared / "AAA.parquet", index=False)

    reads: list[str] = []

    def fake_candidates(symbol):
        return [(symbol, shared / f"{symbol}.parquet")]

    def fake_load(stem):
        reads.append(stem)
        import pandas as pd

        return pd.read_parquet(shared / f"{stem}.parquet")

    import chart_snapshot
    import setup_playbook_study

    monkeypatch.setattr(chart_snapshot, "_daily_store_candidates", fake_candidates)
    monkeypatch.setattr(setup_playbook_study, "_load_daily_frame", fake_load)

    store = D1BarStore(cache_dir=tmp_path / "mirror")
    first = store.load("AAA")
    assert first is not None and len(first) == 20
    assert first.source == "shared"
    assert reads == ["AAA"], "the shared store should be read exactly once"

    # Second call is memory - no store read at all.
    assert store.cached("AAA") is not None
    again = store.load("AAA")
    assert again is not None and reads == ["AAA"]

    # And the mirror landed, so a fresh process skips Drive entirely.
    mirrors = list((tmp_path / "mirror").glob("AAA-*.feather"))
    assert len(mirrors) == 1
    cold = D1BarStore(cache_dir=tmp_path / "mirror")
    warmed = cold.load("AAA")
    assert warmed is not None and warmed.source == "local"
    assert reads == ["AAA"], "the mirror must satisfy a cold start on its own"


def test_a_scanner_update_is_noticed_without_any_manual_invalidation(tmp_path, monkeypatch):
    """The load path itself must notice a changed store - both the memory
    tier and the mirror tier. Nothing in production calls invalidate() when
    the scanner publishes a new session, so a cache that only honors a manual
    invalidate would serve yesterday's bars until process restart."""
    import chart_snapshot
    import setup_playbook_study

    shared = tmp_path / "shared"
    shared.mkdir()
    path = shared / "BBB.parquet"
    _frame(10).to_parquet(path, index=False)

    monkeypatch.setattr(
        chart_snapshot,
        "_daily_store_candidates",
        lambda symbol: [(symbol, shared / f"{symbol}.parquet")],
    )
    monkeypatch.setattr(
        setup_playbook_study,
        "_load_daily_frame",
        lambda stem: __import__("pandas").read_parquet(shared / f"{stem}.parquet"),
    )

    mirror_dir = tmp_path / "mirror"
    store = D1BarStore(cache_dir=mirror_dir)
    assert len(store.load("BBB")) == 10
    assert store.cached("BBB") is not None, "the ten-row series is resident in memory"

    # The scanner appends a session. NO invalidate() call: the next load must
    # see the new mtime past the resident series AND the old-mtime mirror.
    _frame(11).to_parquet(path, index=False)
    os.utime(path, (time.time() + 5, time.time() + 5))
    assert len(store.load("BBB")) == 11
    # Superseded mirrors are pruned, not accumulated.
    assert len(list(mirror_dir.glob("BBB-*.feather"))) == 1

    # And the freshly-loaded series is itself served from memory afterwards.
    assert len(store.load("BBB")) == 11


def test_manual_invalidate_still_drops_a_resident_symbol(tmp_path, monkeypatch):
    import chart_snapshot
    import setup_playbook_study

    shared = tmp_path / "shared"
    shared.mkdir()
    _frame(10).to_parquet(shared / "CCC.parquet", index=False)
    monkeypatch.setattr(
        chart_snapshot,
        "_daily_store_candidates",
        lambda symbol: [(symbol, shared / f"{symbol}.parquet")],
    )
    monkeypatch.setattr(
        setup_playbook_study,
        "_load_daily_frame",
        lambda stem: __import__("pandas").read_parquet(shared / f"{stem}.parquet"),
    )
    store = D1BarStore(cache_dir=tmp_path / "mirror")
    assert store.load("CCC") is not None
    store.invalidate("CCC")
    assert store.cached("CCC") is None


def test_a_directly_put_series_is_authoritative_memory(tmp_path, monkeypatch):
    """Live streaming appends and test fixtures put() series straight into
    memory. Those have no durable-store mtime on record and must be served
    as-is, not re-statted against a store that may not exist."""
    import chart_snapshot

    def _boom(symbol):
        raise AssertionError("a direct put must not trigger store stats")

    monkeypatch.setattr(chart_snapshot, "_daily_store_candidates", _boom)
    store = D1BarStore(cache_dir=tmp_path / "mirror")
    store.put(_series("MEM", rows=7))
    assert len(store.load("MEM")) == 7


def test_lru_evicts_the_least_recently_used_symbol():
    store = D1BarStore(max_symbols=2)
    for symbol in ("AAA", "BBB"):
        store.put(_series(symbol, rows=3))
    assert store.cached("AAA") is not None  # touch AAA, making BBB oldest
    store.put(_series("CCC", rows=3))
    assert store.cached("BBB") is None
    assert store.cached("AAA") is not None and store.cached("CCC") is not None


def test_append_live_bar_ignores_symbols_that_are_not_resident():
    store = D1BarStore()
    bar = {"dt": datetime(2026, 2, 2), "open": 1, "high": 2, "low": 1, "close": 2}
    # Pulling a non-resident symbol in here would turn a streaming update
    # into a blocking store read.
    assert store.append_live_bar("GHOST", bar) is None
    store.put(_series("REAL", rows=3))
    assert len(store.append_live_bar("REAL", bar)) == 4


def test_unknown_symbol_resolves_to_none_not_an_empty_chart(tmp_path, monkeypatch):
    import chart_snapshot
    import setup_playbook_study

    monkeypatch.setattr(
        chart_snapshot,
        "_daily_store_candidates",
        lambda symbol: [(symbol, tmp_path / f"{symbol}.parquet")],
    )
    monkeypatch.setattr(setup_playbook_study, "_load_daily_frame", lambda stem: None)
    store = D1BarStore(cache_dir=tmp_path / "mirror")
    assert store.load("NOPE") is None


# -------------------------------------------------------- ChartDataService
def test_service_delivers_a_snapshot_without_blocking_the_gui_thread():
    if _qt_app() is None:
        pytest.skip("PySide6 unavailable")
    from ui.services.bar_cache import D1BarStore
    from ui.services.chart_data_service import ChartDataService

    store = D1BarStore()
    store.put(_series("NVDA", rows=60))
    service = ChartDataService(store=store)
    got: list[tuple] = []
    service.snapshotReady.connect(lambda *args: got.append(args))

    started = time.perf_counter()
    service.request("NVDA")
    # The request itself must return immediately - that is the whole point.
    assert (time.perf_counter() - started) < 0.05

    assert _pump_until(lambda: bool(got))
    symbol, d1, m5, meta = got[0]
    assert symbol == "NVDA"
    assert len(d1["bars"]) > 0
    assert [o["label"] for o in d1["overlays"]][:3] == ["SMA50", "SMA100", "SMA200"]
    assert m5["bars"] == []  # no intraday bars were supplied
    assert meta["source"] in ("memory", "local", "shared")
    service.shutdown()


def test_a_superseded_request_never_paints_over_a_newer_one():
    if _qt_app() is None:
        pytest.skip("PySide6 unavailable")
    from ui.services.bar_cache import D1BarStore
    from ui.services.chart_data_service import ChartDataService

    store = D1BarStore()
    for symbol in ("AAA", "BBB"):
        store.put(_series(symbol, rows=40))
    service = ChartDataService(store=store)
    delivered: list[str] = []
    service.snapshotReady.connect(lambda symbol, *_: delivered.append(symbol))

    # Two requests for the SAME symbol: only the newest may be delivered.
    service.request("AAA")
    service.request("AAA")
    _pump_until(lambda: len(delivered) >= 1)
    service.wait_for_idle(5000)
    _pump_until(lambda: False, timeout=0.3)
    assert delivered.count("AAA") == 1
    service.shutdown()


def test_last_snapshot_lets_a_revisit_repaint_before_the_rebuild():
    if _qt_app() is None:
        pytest.skip("PySide6 unavailable")
    from ui.services.bar_cache import D1BarStore
    from ui.services.chart_data_service import ChartDataService

    store = D1BarStore()
    store.put(_series("MSFT", rows=40))
    service = ChartDataService(store=store)
    assert service.last_snapshot("MSFT") is None
    service.request("MSFT")
    assert _pump_until(lambda: service.last_snapshot("MSFT") is not None)
    d1, m5 = service.last_snapshot("MSFT")
    assert d1["bars"] and m5 is not None
    service.shutdown()


def test_a_task_finishing_during_shutdown_delivers_nothing():
    """The shutdown race the reviewer called out: a task can pass run()'s
    _closing check, build its snapshot, and only then lose the race with
    shutdown(). Delivery must honor the flag too - emitting into a service
    whose owner is being torn down is the crash shutdown() exists to stop."""
    if _qt_app() is None:
        pytest.skip("PySide6 unavailable")
    from ui.services.bar_cache import D1BarStore
    from ui.services.chart_data_service import ChartDataService

    store = D1BarStore()
    store.put(_series("RACE", rows=10))
    service = ChartDataService(store=store)
    delivered: list[str] = []
    service.snapshotReady.connect(lambda symbol, *_: delivered.append(symbol))
    service.snapshotFailed.connect(lambda symbol: delivered.append(f"failed:{symbol}"))

    d1, m5, meta = service.build_snapshots("RACE", [])
    service._closing = True  # shutdown() won the race after the build finished
    service._finish(1, "RACE", d1, m5, meta)
    _pump_until(lambda: False, timeout=0.2)
    assert delivered == []
    service.shutdown()


def test_prefetch_warms_the_store_off_thread(tmp_path, monkeypatch):
    if _qt_app() is None:
        pytest.skip("PySide6 unavailable")
    import chart_snapshot
    import setup_playbook_study
    from ui.services.chart_data_service import ChartDataService

    shared = tmp_path / "shared"
    shared.mkdir()
    for symbol in ("PPP", "QQQ"):
        _frame(12).to_parquet(shared / f"{symbol}.parquet", index=False)
    monkeypatch.setattr(
        chart_snapshot,
        "_daily_store_candidates",
        lambda symbol: [(symbol, shared / f"{symbol}.parquet")],
    )
    monkeypatch.setattr(
        setup_playbook_study,
        "_load_daily_frame",
        lambda stem: __import__("pandas").read_parquet(shared / f"{stem}.parquet"),
    )

    store = D1BarStore(cache_dir=tmp_path / "mirror")
    service = ChartDataService(store=store)
    done: list[tuple] = []
    service.prefetchFinished.connect(lambda *args: done.append(args))
    assert service.prefetch(["PPP", "QQQ", "PPP"]) == 2  # deduped

    assert _pump_until(lambda: bool(done))
    assert done[0] == (2, 2)
    # Warm means the next chart request needs no store read at all.
    assert store.cached("PPP") is not None and store.cached("QQQ") is not None
    service.shutdown()
