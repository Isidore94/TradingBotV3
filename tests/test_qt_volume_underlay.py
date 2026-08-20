"""D1 volume, drawn as an underlay rather than a stacked sub-plot.

Trader, 2026-08-20: "we need to add volume bars to the D1 charts". A stacked
volume panel is the usual answer and it is the wrong one here - this column is
short enough of vertical space that the capture rail had to move to a tab to
get the candles readable, and a sub-plot would take 20-25% of them straight
back. The columns are drawn in the bottom slice of the price view instead, so
they cost no height at all.

The data was already in hand: the durable daily store carries a volume column,
so nothing here fetches and no IB request is made.
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

_QT = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = _QT.QApplication.instance() or _QT.QApplication([])
    yield app


def _bars(count=40, *, volume=1_000_000.0, start=100.0):
    base = datetime(2026, 7, 1)
    return [
        {
            "dt": base + timedelta(days=index),
            "open": start + index,
            "high": start + index + 1.0,
            "low": start + index - 1.0,
            "close": start + index + 0.5,
            "volume": volume + index,
        }
        for index in range(count)
    ]


def _chart(bars, *, visible=True):
    from ui.widgets.candle_chart import CandleChart

    chart = CandleChart()
    chart.set_volume_visible(visible)
    chart.set_data(bars, [], timeframe="d1")
    return chart


def test_volume_draws_when_the_bars_carry_it():
    chart = _chart(_bars())
    try:
        assert chart.volume_is_drawn()
        assert chart._volume.isVisible()
    finally:
        chart.close()


def test_it_costs_the_candles_no_height():
    """The whole reason it is an underlay: the band sits INSIDE the price
    view, so the y-range is still the candles' and nothing else."""
    bars = _bars()
    chart = _chart(bars)
    try:
        _x, (y_min, y_max) = chart.getPlotItem().vb.viewRange()
        bottom, height = chart._volume._band()
        assert bottom == pytest.approx(y_min)
        # A slice of the view, not an extension of it.
        assert 0 < height < (y_max - y_min)
        assert chart._volume.boundingRect().top() == pytest.approx(y_min)
    finally:
        chart.close()


def test_the_price_range_still_comes_from_the_candles_alone():
    """Volume must never get a vote in the price scale."""
    bars = _bars()
    with_volume = _chart(bars)
    without = _chart([{**bar, "volume": 0.0} for bar in bars])
    try:
        assert with_volume.getPlotItem().vb.viewRange()[1] == pytest.approx(
            without.getPlotItem().vb.viewRange()[1]
        )
    finally:
        with_volume.close()
        without.close()


def test_no_volume_draws_nothing_rather_than_a_flat_row_of_zero():
    """Missing data is uncertainty, never confirmation (plan.md sec 5). A row
    of empty columns would read as "no volume traded"."""
    chart = _chart([{**bar, "volume": 0.0} for bar in _bars()])
    try:
        assert not chart.volume_is_drawn()
        assert not chart._volume.isVisible()
        assert chart._volume.boundingRect().isEmpty()
    finally:
        chart.close()


def test_a_missing_volume_key_is_survivable():
    bars = [{k: v for k, v in bar.items() if k != "volume"} for bar in _bars()]
    chart = _chart(bars)
    try:
        assert not chart.volume_is_drawn()
    finally:
        chart.close()


def test_unparseable_volume_never_raises_mid_render():
    bars = _bars()
    bars[3]["volume"] = "not a number"
    bars[7]["volume"] = None
    bars[9]["volume"] = float("nan")
    chart = _chart(bars)
    try:
        assert chart.volume_is_drawn(), "the readable bars still draw"
    finally:
        chart.close()


def test_turning_it_off_leaves_the_chart_alone():
    bars = _bars()
    chart = _chart(bars, visible=False)
    try:
        assert not chart.volume_is_drawn()
        assert not chart._volume.isVisible()
        chart.set_volume_visible(True)
        assert chart.volume_is_drawn()
    finally:
        chart.close()


def test_a_pan_is_a_transform_not_a_re_render():
    """The band is defined against the view, so moving the view must not cost
    a re-record of every column."""
    chart = _chart(_bars())
    try:
        before = chart._volume._picture
        _x, (y_min, y_max) = chart.getPlotItem().vb.viewRange()
        span = y_max - y_min
        chart.getPlotItem().vb.setYRange(y_min + span, y_max + span, padding=0)
        assert chart._volume._picture is before, "the picture was re-recorded"
        bottom, _height = chart._volume._band()
        assert bottom > y_min, "but the band followed the view"
    finally:
        chart.close()


def test_the_d1_chart_asks_for_it_and_the_m5_does_not():
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    widget = SymbolSnapshotWidget()
    try:
        assert widget.d1_chart._show_volume is True
        assert widget.m5_chart._show_volume is False
    finally:
        widget.deleteLater()


def test_drawing_volume_fetches_nothing():
    """No network, no IB request: the bars handed in already carry it."""
    import chart_snapshot

    def _boom(*_a, **_k):
        raise AssertionError("volume rendering must not fetch")

    original = getattr(chart_snapshot, "load_d1_bars")
    chart_snapshot.load_d1_bars = _boom
    try:
        chart = _chart(_bars())
        assert chart.volume_is_drawn()
        chart.close()
    finally:
        chart_snapshot.load_d1_bars = original
