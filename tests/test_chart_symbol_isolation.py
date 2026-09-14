"""CH-SYM: a reused chart must never show the last symbol's snapshot.

The centre review pane keeps one real ``SymbolSnapshotWidget`` while the
trader moves through rows.  These tests exercise that widget and its real
offscreen CandleCharts through ``set_symbol``; the tiny store and bot only
stand in for the documented cached-data seams.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt


def _bar(stamp: datetime, close: float) -> dict:
    """A cached M5 row in the shape BounceBot exposes to the chart."""
    return {
        "dt": stamp,
        "open": close - 0.10,
        "high": close + 0.20,
        "low": close - 0.30,
        "close": close,
        "volume": 10_000.0,
    }


def _daily(close: float) -> list[dict]:
    """Hand-pinned durable rows; the widget only needs a drawable D1 cache."""
    return [
        _bar(datetime(2026, 9, 8) + timedelta(days=offset), close + offset)
        for offset in range(3)
    ]


class _Series:
    source = "shared"

    def __init__(self, bars: list[dict]) -> None:
        self._bars = list(bars)

    def as_bar_dicts(self) -> list[dict]:
        return [dict(bar) for bar in self._bars]

    def __len__(self) -> int:
        return len(self._bars)


class _Store:
    def __init__(self) -> None:
        self._rows = {"AAA": _daily(100.0)}

    def load(self, symbol: str):
        rows = self._rows.get(str(symbol).upper())
        return _Series(rows) if rows else None

    cached = load

    def prefetch(self, _symbols) -> int:
        return 0


class _Bot:
    """The in-memory ``m5_chart_bars`` seam, with symbol-distinct prices."""

    def __init__(self, *, aaa_close: float = 101.0, bbb_close: float = 201.0) -> None:
        start = datetime(2026, 9, 11, 9, 30)
        self.rows = {
            "AAA": [
                _bar(start, aaa_close),
                _bar(start + timedelta(minutes=5), aaa_close + 1.0),
            ],
            # Deliberately later than AAA: the old merge incorrectly prepends
            # AAA because its timestamps fall before this new chunk.
            "BBB": [_bar(start + timedelta(minutes=30), bbb_close)],
        }
        self.raise_for: set[str] = set()

    def m5_chart_bars(self, symbol: str, max_sessions: int = 2) -> list[dict]:
        symbol = str(symbol).upper()
        if symbol in self.raise_for:
            raise RuntimeError(f"{symbol} cache read failed")
        return [dict(bar) for bar in self.rows.get(symbol, [])]


@pytest.fixture
def qapp():
    pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def chart(qapp, monkeypatch):
    from ui.services import chart_bar_refresh
    from ui.services import chart_data_service as service_mod
    from ui.widgets import symbol_snapshot_dialog as mod

    monkeypatch.setattr(service_mod, "shared_store", _Store)
    monkeypatch.setattr(
        chart_bar_refresh,
        "shared_refresh_service",
        lambda: type("NoRefresh", (), {"best_bars": lambda _self, _symbol, bars: bars})(),
    )
    # Provider recovery is not part of a symbol handoff, and the tests must
    # stay fully offline.  The regular snapshot worker still builds and draws.
    monkeypatch.setattr(mod.SymbolSnapshotWidget, "_start_d1_backfill", lambda *_: None)
    monkeypatch.setattr(mod.SymbolSnapshotWidget, "_start_forming_fetch", lambda *_: None)

    widget = mod.SymbolSnapshotWidget(compact=True)
    widget.resize(900, 700)
    yield widget
    widget._data.shutdown()
    widget.deleteLater()
    qapp.processEvents()


def _drain(widget, qapp) -> None:
    for _round in range(8):
        widget._data.wait_for_idle(5000)
        qapp.processEvents()


def _close_values(widget) -> list[float]:
    return [float(bar["close"]) for bar in widget.cached_m5_bars()]


def test_switching_to_a_pending_symbol_clears_old_chart_quick_fill_and_level_alert(
    chart, qapp
):
    """A pending BBB must not offer an AAA price or alert anchor to click."""
    bot = _Bot()
    chart.set_symbol("AAA", bot=bot)
    _drain(chart, qapp)
    assert chart.d1_chart.bar_count() > 0
    assert chart.m5_chart.bar_count() == 2
    assert chart.quick_fill("last") == pytest.approx(102.0)

    emitted = []
    chart.d1LevelAlertRequested.connect(lambda *row: emitted.append(row))
    bot.rows["BBB"] = []
    chart.set_symbol("BBB", bot=bot)

    assert chart.d1_chart.bar_count() == 0
    assert chart.m5_chart.bar_count() == 0
    assert chart.cached_m5_bars() == []
    assert chart.quick_fill("last") is None
    chart.request_d1_level_alert("above", 0)
    chart.request_m5_level_alert("above", 0)
    assert emitted == []

    # The worker must receive BBB's empty cache too.  Before the reset it
    # receives retained AAA M5 rows and synthesises an AAA-looking B preview.
    _drain(chart, qapp)
    assert chart.d1_chart.bar_count() == 0
    assert chart.m5_chart.bar_count() == 0


@pytest.mark.parametrize(
    ("aaa_close", "bbb_close"),
    [(101.0, 201.0), (301.0, 201.0)],
    ids=("old_below_new", "old_above_new"),
)
def test_fresh_b_symbol_m5_never_prepends_the_previous_symbols_history(
    chart, qapp, aaa_close, bbb_close
):
    """The real refresh path may retain BBB history, but never AAA history."""
    bot = _Bot(aaa_close=aaa_close, bbb_close=bbb_close)
    chart.set_symbol("AAA", bot=bot)
    _drain(chart, qapp)
    assert _close_values(chart) == [aaa_close, aaa_close + 1.0]

    chart.set_symbol("BBB", bot=bot)
    _drain(chart, qapp)

    assert _close_values(chart) == [bbb_close]
    assert chart.m5_chart.bar_count() == 1


def test_a_b_cache_read_error_after_switch_cannot_restore_a_bars(chart, qapp):
    """The failure fallback is local to the current symbol's drawn cache."""
    bot = _Bot()
    chart.set_symbol("AAA", bot=bot)
    _drain(chart, qapp)
    assert _close_values(chart) == [101.0, 102.0]

    bot.raise_for.add("BBB")
    chart.set_symbol("BBB", bot=bot)

    assert chart.cached_m5_bars() == []
    assert chart.m5_chart.bar_count() == 0


def test_refreshing_the_same_symbol_keeps_its_chart_and_a_cached_b_snapshot_is_drawn(
    chart, qapp
):
    """Isolation must not throw away valid data for the symbol still in view."""
    bot = _Bot()
    chart.set_symbol("AAA", bot=bot)
    _drain(chart, qapp)
    chart.set_symbol("AAA", bot=bot)
    _drain(chart, qapp)
    assert _close_values(chart) == [101.0, 102.0]

    chart.set_symbol("BBB", bot=bot)
    _drain(chart, qapp)
    chart.set_symbol("AAA", bot=bot)
    _drain(chart, qapp)
    chart.set_symbol("BBB", bot=bot)

    assert _close_values(chart) == [201.0]
