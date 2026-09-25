"""TJ-2A: durable, completed M5 session bars for Day Review.

These are packet tests, deliberately written before ``day_review_bars`` exists.
They pin the durable file and worker seams rather than a yfinance implementation;
the injected downloader is the only permitted test data source.
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SESSION = "2026-09-10"
OPEN_SESSION = "2026-09-18"
PACIFIC = ZoneInfo("America/Los_Angeles")
EASTERN = ZoneInfo("America/New_York")


def _bars_module():
    """Import inside each test so missing production code fails each assertion."""
    return importlib.import_module("day_review_bars")


def _bar(hour: int, minute: int, *, close: float = 101.0) -> dict:
    return {
        "dt": datetime(2026, 9, 10, hour, minute, tzinfo=PACIFIC),
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": close,
        "volume": 1234,
    }


def test_decided_symbols_unions_recap_trade_names_and_all_benchmarks(monkeypatch):
    bars = _bars_module()

    class _Journal:
        def list_trades(self, *, trade_date):
            assert trade_date == SESSION
            return [{"symbol": "NVDA"}, {"symbol": "AAPL"}, {"symbol": ""}]

    monkeypatch.setattr(
        bars.daily_recap_reader,
        "_decisions",
        lambda *_a, **_k: ({"symbol": "TSLA"}, {"symbol": "AAPL"}),
    )
    monkeypatch.setattr(bars, "shared_journal_service", lambda: _Journal())

    names = bars.decided_symbols(SESSION, sources=object())

    # TJ-14A item 6: the internals the Trade Mentor watches join the ONE
    # batched post-close download, so an hour the trader never answered can
    # still have its internals rebuilt from the durable tape. The union is
    # WIDER, and it is still exactly the decisions, the trades, the benchmarks
    # and that fixed list - nothing else.
    from trade_mentor_context import SYMBOLS as INTERNALS

    assert names == {"TSLA", "AAPL", "NVDA", "SPY", "QQQ", "IWM", "VXX"} | set(INTERNALS)
    assert set(bars.BENCHMARKS) <= names
    assert set(INTERNALS) <= names


def test_fetch_uses_one_duplicate_free_batched_call_per_fifty_and_omits_missing_symbol():
    bars = _bars_module()
    symbols = [f"S{i:02d}" for i in range(51)] + ["S00", "MISSING"]
    calls: list[tuple[tuple, dict]] = []

    def fake_downloader(*args, **kwargs):
        calls.append((args, kwargs))
        tickers = tuple(kwargs.get("tickers", args[0]))
        return {symbol: [_bar(9, 30)] for symbol in tickers if symbol != "MISSING"}

    result = bars.fetch_session_bars(symbols, SESSION, downloader=fake_downloader)

    assert len(calls) == 2
    requested = []
    for args, kwargs in calls:
        tickers = tuple(kwargs.get("tickers", args[0]))
        assert len(tickers) <= 50
        requested.extend(tickers)
        assert kwargs["interval"] == "5m"
        assert kwargs["auto_adjust"] is False
        assert kwargs["prepost"] is False
    assert len(requested) == len(set(requested))
    assert set(requested) == set(symbols)
    assert "MISSING" not in result
    assert result["S00"] == [_bar(9, 30)]


def test_injected_downloader_means_pytest_never_calls_real_yfinance(monkeypatch):
    bars = _bars_module()
    real_calls: list[object] = []

    def no_real_download(*_args, **_kwargs):
        real_calls.append(True)
        raise AssertionError("pytest attempted a real yfinance download")

    import yfinance

    monkeypatch.setattr(yfinance, "download", no_real_download)
    answer = bars.fetch_session_bars(
        ["SPY"], SESSION, downloader=lambda *_a, **_k: {"SPY": [_bar(9, 30)]}
    )

    assert answer == {"SPY": [_bar(9, 30)]}
    assert real_calls == []


def test_exchange_open_bar_survives_and_is_stored_market_local():
    bars = _bars_module()
    eastern_open = {
        **_bar(9, 30),
        "dt": datetime(2026, 9, 10, 9, 30, tzinfo=EASTERN),
    }

    answer = bars.fetch_session_bars(
        ["SPY"], SESSION, downloader=lambda *_a, **_k: {"SPY": [eastern_open]}
    )

    stamp = answer["SPY"][0]["dt"]
    assert stamp == datetime(2026, 9, 10, 6, 30, tzinfo=PACIFIC)
    assert stamp.tzinfo == PACIFIC


def test_parquet_round_trip_preserves_each_bar_and_attached_market_zone(monkeypatch, tmp_path):
    bars = _bars_module()
    monkeypatch.setattr(bars, "DAY_REVIEW_DIR", tmp_path)
    monkeypatch.setattr(bars, "session_is_closed", lambda *_a, **_k: True)
    written = {"SPY": [_bar(9, 30), _bar(9, 35, close=102.5)], "NVDA": [_bar(9, 30)]}

    path = bars.write_session_bars(SESSION, written)

    assert path == bars.bars_path(SESSION)
    assert path.suffix == ".parquet"
    assert path.is_file()
    assert bars.read_session_bars(SESSION) == written
    assert all(
        row["dt"].tzinfo is not None
        and row["dt"].utcoffset() == PACIFIC.utcoffset(row["dt"])
        for rows in bars.read_session_bars(SESSION).values()
        for row in rows
    )


def test_write_refuses_an_open_session_before_any_file_is_written(monkeypatch, tmp_path):
    bars = _bars_module()
    monkeypatch.setattr(bars, "DAY_REVIEW_DIR", tmp_path)
    monkeypatch.setattr(bars, "session_is_closed", lambda *_a, **_k: False)

    with pytest.raises(ValueError, match="closed"):
        bars.write_session_bars(OPEN_SESSION, {"SPY": [_bar(9, 30)]})

    assert not list(tmp_path.rglob("*.parquet"))


def test_write_deduplicates_the_symbol_timestamp_grain(monkeypatch, tmp_path):
    bars = _bars_module()
    monkeypatch.setattr(bars, "DAY_REVIEW_DIR", tmp_path)
    monkeypatch.setattr(bars, "session_is_closed", lambda *_a, **_k: True)
    duplicate = _bar(9, 30)

    bars.write_session_bars(SESSION, {"SPY": [duplicate, dict(duplicate)]})

    assert bars.read_session_bars(SESSION) == {"SPY": [duplicate]}


def test_closed_day_read_uses_spy_from_the_session_file_not_live_cache(monkeypatch):
    bars = _bars_module()
    from ui.services.day_review_service import DayReviewService

    stored = [_bar(9, 30)]
    monkeypatch.setattr(bars, "read_session_bars", lambda session: {"SPY": stored})
    monkeypatch.setattr(bars, "session_is_closed", lambda *_a, **_k: True)
    service = DayReviewService()
    monkeypatch.setattr(service, "_read_recap", lambda *_a, **_k: object())
    monkeypatch.setattr(service, "_trades", lambda *_a, **_k: [])

    payload = service.read_day(SESSION, now=datetime(2026, 9, 11, 7, 30), spy_m5_bars=[])

    assert payload["spy_m5_bars"] == stored


pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the worker seam is a Qt panel")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_post_close_worker_fetches_after_index_off_the_timer_thread(qapp, monkeypatch):
    """The index remains first; bars are a non-fatal second operation on its worker."""
    _bars_module()
    from ui.panels.day_review_panel import _IndexBuildWorker

    calls: list[tuple[str, int]] = []

    class _Service:
        def build_index_for(self, session_date, **_kwargs):
            calls.append(("index", threading.get_ident()))

        def build_session_bars_for(self, session_date, **_kwargs):
            calls.append(("bars", threading.get_ident()))

    worker = _IndexBuildWorker(_Service(), SESSION)
    worker.start()
    assert worker.wait(5000), "the post-close worker did not finish"

    assert [name for name, _thread in calls] == ["index", "bars"]
    assert all(thread != threading.get_ident() for _name, thread in calls)


def test_open_session_never_starts_a_bars_fetch_from_the_page_timer(qapp, monkeypatch):
    """An in-progress session can use the live hand-off only; it never downloads bars."""
    _bars_module()
    from ui.panels.day_review_panel import DayReviewPanel

    class _Service:
        def __init__(self):
            self.bar_calls = []

        def read_day(self, session_date, **_kwargs):
            return {"session_date": session_date}

        def build_index_for(self, *_a, **_k):
            raise AssertionError("the open-session timer tried to build an index")

        def build_session_bars_for(self, session_date, **_kwargs):
            self.bar_calls.append(session_date)

    service = _Service()
    panel = DayReviewPanel(service=service, clock=lambda: datetime(2026, 9, 18, 10, 0))
    try:
        monkeypatch.setattr(panel, "reload", lambda: None)
        monkeypatch.setattr(panel, "show_session", lambda _session: None)
        import daily_recap_schedule

        monkeypatch.setattr(daily_recap_schedule, "due_session", lambda *_a, **_k: OPEN_SESSION)
        monkeypatch.setattr(daily_recap_schedule, "post_close_due_session", lambda *_a, **_k: None)
        assert panel.poll_auto_read() == OPEN_SESSION
        assert service.bar_calls == []
    finally:
        panel.shutdown()
        panel.deleteLater()


def test_missing_past_file_starts_one_backfill_worker_and_says_what_is_happening(
    qapp, monkeypatch
):
    """Past-file recovery is never a Qt-thread/network action or a double fetch."""
    bars = _bars_module()
    from ui.panels.day_review_panel import DayReviewPanel

    started = threading.Event()

    class _Service:
        def __init__(self):
            self.calls: list[tuple[str, int]] = []

        def read_day(self, session_date, **_kwargs):
            return {"session_date": session_date}

        def backfill_session_bars_for(self, session_date, **_kwargs):
            self.calls.append((str(session_date), threading.get_ident()))
            started.set()
            time.sleep(0.15)

    service = _Service()
    monkeypatch.setattr(bars, "read_session_bars", lambda _session: None)
    monkeypatch.setattr(bars, "session_is_closed", lambda *_a, **_k: True)
    panel = DayReviewPanel(service=service, clock=lambda: datetime(2026, 9, 11, 7, 30))
    try:
        gui_thread = threading.get_ident()
        panel._backfill_bars_for(SESSION)
        assert started.wait(2.0), "the missing past session never started a backfill"
        # A second open while the first fetch is live must coalesce, not hit
        # yfinance twice for the same file.
        panel._backfill_bars_for(SESSION)
        assert "fetching" in panel.status.text().lower()
        assert SESSION in panel.status.text()
        worker = panel._bars_worker
        assert worker is not None and worker.wait(5000)
        qapp.processEvents()
        assert service.calls == [(SESSION, service.calls[0][1])]
        assert service.calls[0][1] != gui_thread
    finally:
        panel.shutdown()
        panel.deleteLater()


def test_backfill_refuses_the_open_session(qapp, monkeypatch):
    """The backfill exception is for a PAST session only, never the live tape."""
    bars = _bars_module()
    from ui.panels.day_review_panel import DayReviewPanel

    class _Service:
        def __init__(self):
            self.calls = []

        def read_day(self, session_date, **_kwargs):
            return {"session_date": session_date}

        def backfill_session_bars_for(self, session_date, **_kwargs):
            self.calls.append(session_date)

    service = _Service()
    monkeypatch.setattr(bars, "read_session_bars", lambda _session: None)
    monkeypatch.setattr(bars, "session_is_closed", lambda *_a, **_k: False)
    panel = DayReviewPanel(service=service, clock=lambda: datetime(2026, 9, 18, 10, 0))
    try:
        panel._backfill_bars_for(OPEN_SESSION)
        assert service.calls == []
        assert panel._bars_worker is None
    finally:
        panel.shutdown()
        panel.deleteLater()


def test_operations_audit_reads_count_from_its_injected_bars_root(monkeypatch, tmp_path):
    """System Health must not resolve the process-global production bars root."""
    bars = _bars_module()
    import operations_audit

    monkeypatch.setattr(bars, "DAY_REVIEW_DIR", tmp_path)
    monkeypatch.setattr(bars, "session_is_closed", lambda *_a, **_k: True)
    bars.write_session_bars(SESSION, {"SPY": [_bar(9, 30)], "QQQ": [_bar(9, 30)]})
    # If the audit reader ignores its root, this empty directory produces no
    # count even though its injected root contains two symbols.
    monkeypatch.setattr(bars, "DAY_REVIEW_DIR", tmp_path / "wrong-production-root")

    check = operations_audit._day_review_bars_check(tmp_path)

    assert check["id"] == "day_review_bars"
    assert SESSION in check["summary"]
    assert "2" in check["summary"]
    assert check["details"] == {"last_session": SESSION, "symbol_count": 2}
