"""The D1 bar dicts the Alert Center polls are built off the Qt thread and reused.

2026-09-30: the 60 s D1 poll rebuilt ~490 dicts per symbol on the Qt thread
once a minute (240-350 s a day, one 1 s stall per minute) because each
prefetch stored a NEW series object and the memo compared by identity only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")

from ui.services.bar_cache import BarSeries  # noqa: E402
from ui.services.chart_data_service import ChartDataService, _PrefetchTask  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _series(symbol: str, closes, *, start="2026-01-01") -> BarSeries:
    n = len(closes)
    dt = np.arange(np.datetime64(start, "ns"), np.datetime64(start, "ns") + np.timedelta64(n, "D"), np.timedelta64(1, "D"))
    arr = np.asarray(closes, dtype=float)
    return BarSeries(symbol, dt, arr, arr + 1, arr - 1, arr, np.full(n, 1000.0), source="memory")


class _Store:
    def __init__(self):
        self.series: dict[str, BarSeries] = {}
        self.prefetched: list[list[str]] = []

    def cached(self, symbol):
        return self.series.get(symbol)

    def prefetch(self, symbols):
        self.prefetched.append(list(symbols))
        return len(symbols)


@pytest.fixture
def service(qapp):
    store = _Store()
    svc = ChartDataService(store=store, max_threads=1)
    try:
        yield svc, store
    finally:
        svc.shutdown()


def test_same_bars_in_a_new_series_object_reuse_the_dicts(service, monkeypatch):
    svc, store = service
    store.series["NVDA"] = _series("NVDA", [1.0, 2.0, 3.0])
    first = svc.cached_bar_dicts("NVDA")
    built: list[str] = []
    monkeypatch.setattr(BarSeries, "as_bar_dicts", lambda self: built.append(self.symbol) or [])
    store.series["NVDA"] = _series("NVDA", [1.0, 2.0, 3.0])  # a refresh that changed nothing
    assert svc.cached_bar_dicts("NVDA") is first
    assert built == []


def test_changed_bars_rebuild(service):
    svc, store = service
    store.series["NVDA"] = _series("NVDA", [1.0, 2.0, 3.0])
    first = svc.cached_bar_dicts("NVDA")
    store.series["NVDA"] = _series("NVDA", [1.0, 2.0, 3.5])
    second = svc.cached_bar_dicts("NVDA")
    assert second is not first
    assert second[-1]["close"] == 3.5


def test_the_prefetch_worker_materializes_the_dicts(service, monkeypatch):
    svc, store = service
    store.series["AMD"] = _series("AMD", [5.0, 6.0])
    monkeypatch.setattr(svc, "_cache_earnings_anchor_from_source", lambda symbol: None)
    _PrefetchTask(svc, ["AMD"]).run()  # the worker body, run inline
    built: list[str] = []
    monkeypatch.setattr(BarSeries, "as_bar_dicts", lambda self: built.append(self.symbol) or [])
    assert len(svc.cached_bar_dicts("AMD")) == 2
    assert built == []
