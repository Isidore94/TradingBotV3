"""The widget half: a corrupt bar cannot paint over the session.

``test_bar_integrity.py`` covers the judgement and the diagnostic. This covers
what the trader actually sees - the chart still ranges on the real bars, the
bad one is drawn as a dashed outline rather than a solid wall, and the chart
says out loud that it did that.
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

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from ui import bar_integrity  # noqa: E402
from ui.widgets.candle_chart import CandleChart  # noqa: E402


def _bar(minute: int, open_: float, high: float, low: float, close: float):
    return {
        "dt": datetime(2026, 8, 21, 6, 30) + timedelta(minutes=5 * minute),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000.0,
    }


def _session() -> list[dict]:
    return [_bar(i, 47.50, 47.60, 47.40, 47.55) for i in range(12)]


def _visible_prices(chart: CandleChart) -> tuple[float, float]:
    """The view's y-range back in price space.

    The desk's charts default to log10 scaling, so viewRange() is in log
    space; a test that forgot would compare 1.67 against 47.4 and pass or
    fail for the wrong reason.
    """
    (_x0, _x1), (low, high) = chart.getPlotItem().vb.viewRange()
    if getattr(chart, "_log_active", False):
        return (10.0 ** low, 10.0 ** high)
    return (low, high)


def test_a_bad_close_does_not_stretch_the_view(qtbot=None):
    """The GFS shape: a close from nowhere, on an otherwise normal session."""
    chart = CandleChart()
    chart.set_data(_session() + [_bar(12, 47.50, 47.62, 47.40, 48.62)], timeframe="m5")
    y_low, y_high = _visible_prices(chart)
    # Padding is 5% of the span; the bogus 48.62 must be nowhere near it.
    assert y_high < 47.8, f"the bad close stretched the view to {y_high}"
    assert y_low > 47.2


def test_the_bad_bar_is_reported_not_hidden():
    chart = CandleChart()
    chart.set_data(_session() + [_bar(12, 0.0, 47.62, 47.40, 47.55)], timeframe="m5")
    defects = chart.bar_defects()
    assert [d.index for d in defects] == [12]
    assert defects[0].defect == bar_integrity.DEFECT_OPEN_OUTSIDE
    assert chart._bad_bar_note is not None and chart._bad_bar_note.isVisible()


def test_a_healthy_series_shows_no_note():
    chart = CandleChart()
    chart.set_data(_session(), timeframe="m5")
    assert chart.bar_defects() == []
    assert chart._bad_bar_note is None or not chart._bad_bar_note.isVisible()


def test_a_bar_missing_a_price_does_not_crash_the_chart():
    """Before the guard this raised KeyError straight out of set_data."""
    bars = _session()
    broken = _bar(12, 47.5, 47.6, 47.4, 47.55)
    del broken["low"]
    chart = CandleChart()
    chart.set_data(bars + [broken], timeframe="m5")
    assert [d.defect for d in chart.bar_defects()] == [bar_integrity.DEFECT_NOT_NUMERIC]
    # Nothing about that bar's geometry was usable, so it is counted and not drawn.
    assert not chart.bar_defects()[0].drawable


def test_an_all_bad_series_still_draws_the_rest_of_the_chart():
    chart = CandleChart()
    chart.set_data([_bar(0, 0.0, 47.6, 47.4, 48.9), _bar(1, 0.0, 47.8, 47.3, 49.0)], timeframe="m5")
    y_low, y_high = _visible_prices(chart)
    assert y_high < 48.2 and y_low > 47.0
