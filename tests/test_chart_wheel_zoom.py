"""Scrolling back on a chart shows MORE of it (trader, 2026-09-21).

The trader's sentence is *"when i scroll back on the visual charts i want things
to get smaller so i can see more, right now it just moves the chart to the left
and squishes things"*. Two causes, both measured in the widget:

* the price scale was taken ONCE, from the window the chart opened on, so every
  older candle that came into view was drawn against today's prices - off the
  top or the bottom, never smaller;
* the wheel zoomed around the cursor, so with the newest candle on screen half
  of every zoom-out went into the empty space to the RIGHT of it and the whole
  chart slid left.

These tests drive the real ``CandleChart`` offscreen with a real ``QWheelEvent``
and the WS-CH golden fixture's bars (a walk from ~39 to ~197, so a scale taken
from the wrong window is a number, not a rounding).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

GOLDEN_PATH = ROOT_DIR / "tests" / "fixtures" / "ws_ch_chart_history_v1.json"
VISIBLE_SESSIONS = 90
#: One notch of a mouse wheel, in Qt's eighths of a degree.
NOTCH = 120


def _bars(count: int = 1000) -> list[dict]:
    rows = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["bars"][-count:]
    return [
        {
            "dt": datetime.fromisoformat(day),
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
        }
        for day, open_, high, low, close, volume in rows
    ]


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def chart(qapp):
    from ui.widgets.candle_chart import CandleChart

    widget = CandleChart()
    widget.resize(900, 420)
    widget.show()
    qapp.processEvents()
    widget.set_data(_bars(), [], timeframe="d1", initial_view_sessions=VISIBLE_SESSIONS)
    qapp.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def _x_range(chart) -> tuple[float, float]:
    (low, high), _y = chart.getPlotItem().vb.viewRange()
    return float(low), float(high)


def _visible_prices(chart) -> tuple[float, float]:
    _x, (low, high) = chart.getPlotItem().vb.viewRange()
    return chart.price_at(float(low)), chart.price_at(float(high))


def _bars_on_screen(chart) -> list[dict]:
    low, high = _x_range(chart)
    first = max(0, int(low) + (0 if low == int(low) else 1))
    last = min(chart.bar_count() - 1, int(high))
    return [chart.bar_at(index) for index in range(first, last + 1)]


def _wheel(chart, qapp, notches: int, *, at_fraction: float = 0.5) -> None:
    """Turn the wheel over the chart. Negative notches = scrolling BACK."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    position = QPointF(chart.width() * at_fraction, chart.height() * 0.5)
    event = QWheelEvent(
        position,
        QPointF(chart.mapToGlobal(position.toPoint())),
        QPoint(0, 0),
        QPoint(0, NOTCH * notches),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    # Through the viewport, the door a real mouse uses - calling the override
    # directly would still pass if Qt never routed a wheel to it.
    qapp.sendEvent(chart.viewport(), event)
    qapp.processEvents()


def _x_under_cursor(chart, at_fraction: float) -> float:
    from PySide6.QtCore import QPoint

    point = QPoint(int(chart.width() * at_fraction), int(chart.height() * 0.5))
    return float(chart.getPlotItem().vb.mapSceneToView(chart.mapToScene(point)).x())


@pytest.mark.qt
def test_scrolling_back_keeps_the_newest_candle_where_it_was_and_shows_older_ones(chart, qapp):
    """The chart does not slide left: the zoom-out goes into the PAST."""
    low_before, high_before = _x_range(chart)

    _wheel(chart, qapp, -3)

    low_after, high_after = _x_range(chart)
    assert high_after == pytest.approx(high_before, abs=0.5), (
        "the newest candle moved - the zoom-out was spent on empty space to its right"
    )
    assert low_after < low_before - 50, "scrolling back did not bring older candles in"


@pytest.mark.qt
def test_scrolling_back_makes_the_candles_smaller_so_every_one_on_screen_fits(chart, qapp):
    """The trader's 'get smaller so i can see more': the price scale follows."""
    low_price_before, high_price_before = _visible_prices(chart)

    for _ in range(8):
        _wheel(chart, qapp, -3)

    shown = _bars_on_screen(chart)
    assert len(shown) > 600, "eight scrolls back should reach years of candles"
    lowest = min(bar["low"] for bar in shown)
    highest = max(bar["high"] for bar in shown)
    low_price, high_price = _visible_prices(chart)
    assert lowest < low_price_before, "the fixture's older candles sit under the opening scale"
    assert low_price <= lowest, f"candles down to {lowest:.2f} are cut off at {low_price:.2f}"
    assert high_price >= highest
    # ...and it is a FIT, not a scale thrown wide open.
    assert low_price == pytest.approx(lowest, rel=0.25)
    assert high_price == pytest.approx(highest, rel=0.25)
    assert (high_price - low_price) > (high_price_before - low_price_before)


@pytest.mark.qt
def test_scrolling_forward_again_brings_the_recent_candles_back_up_to_size(chart, qapp):
    low_before, high_before = _x_range(chart)
    prices_before = _visible_prices(chart)

    _wheel(chart, qapp, -3)
    _wheel(chart, qapp, 3)

    low_after, high_after = _x_range(chart)
    assert high_after == pytest.approx(high_before, abs=0.5)
    assert low_after == pytest.approx(low_before, abs=1.5)
    assert _visible_prices(chart) == pytest.approx(prices_before, rel=0.02)


@pytest.mark.qt
def test_scrolling_back_stops_at_the_oldest_candle(chart, qapp):
    """Past the oldest bar there is nothing to see, only squeezing."""
    for _ in range(40):
        _wheel(chart, qapp, -3)

    low, high = _x_range(chart)
    assert low >= -1.0 - 0.02 * chart.bar_count()
    assert high <= chart.bar_count() + 0.02 * chart.bar_count()


@pytest.mark.qt
def test_scrolling_in_cannot_squeeze_the_chart_down_to_nothing(chart, qapp):
    for _ in range(40):
        _wheel(chart, qapp, 3)

    low, high = _x_range(chart)
    assert high - low >= 5.0


@pytest.mark.qt
def test_away_from_the_newest_candle_the_wheel_zooms_around_the_cursor(chart, qapp):
    """Looking at 2023, the trader zooms on what is under the mouse."""
    chart.getPlotItem().setXRange(300.0, 400.0, padding=0)
    qapp.processEvents()
    under_before = _x_under_cursor(chart, 0.25)

    _wheel(chart, qapp, -2, at_fraction=0.25)

    low, high = _x_range(chart)
    assert high - low > 100.0
    assert high < chart.bar_count() - 1, "it jumped to the newest candle"
    assert _x_under_cursor(chart, 0.25) == pytest.approx(under_before, abs=1.0)


@pytest.mark.qt
def test_dragging_back_through_the_chart_refits_the_price_scale(chart, qapp):
    """A drag is the other way to 'scroll back'; the same candles, the same rule."""
    view = chart.getPlotItem().vb
    view.translateBy(x=-600.0)
    view.sigRangeChangedManually.emit(view.state["mouseEnabled"])
    qapp.processEvents()

    shown = _bars_on_screen(chart)
    assert shown
    lowest = min(bar["low"] for bar in shown)
    highest = max(bar["high"] for bar in shown)
    low_price, high_price = _visible_prices(chart)
    assert low_price <= lowest and high_price >= highest
    assert high_price == pytest.approx(highest, rel=0.25), (
        f"the scale still reaches {high_price:.2f} for candles that top out at {highest:.2f}"
    )


@pytest.mark.qt
def test_a_malformed_candle_coming_into_view_gets_no_vote_on_the_scale(chart, qapp):
    """The bar-integrity rule holds while zooming, not only when the chart opens."""
    bars = _bars()
    bars[700]["high"] = 5000.0
    bars[700]["close"] = 9000.0  # close outside the range: drawable, never trusted
    chart.set_data(bars, [], timeframe="d1", initial_view_sessions=VISIBLE_SESSIONS)
    qapp.processEvents()

    for _ in range(8):
        _wheel(chart, qapp, -3)

    assert any(bar["high"] == 5000.0 for bar in _bars_on_screen(chart))
    _low_price, high_price = _visible_prices(chart)
    assert high_price < 1000.0, "a broken candle set the price scale"


@pytest.mark.qt
def test_the_wheel_does_nothing_on_an_empty_chart(qapp):
    from ui.widgets.candle_chart import CandleChart

    widget = CandleChart()
    widget.resize(600, 300)
    widget.show()
    qapp.processEvents()
    try:
        widget.set_data([], [], timeframe="m5")
        _wheel(widget, qapp, -3)
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()
