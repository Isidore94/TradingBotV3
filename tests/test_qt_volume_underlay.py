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


def _chart(bars, *, visible=True, earnings=False):
    from ui.widgets.candle_chart import CandleChart

    chart = CandleChart()
    chart.set_volume_visible(visible)
    chart.set_earnings_visible(earnings)
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


# --------------------------------------------------------------------------
# Earnings ribbon (trader, 2026-08-20): "an E on the earnings candles. at the
# top a projection of the next earnings candle."
#
# Trader chose: E on a top ribbon aligned to its candle, and the projection
# PINNED at top-right with no axis change - the projection sits a median 48
# sessions past the last bar, so drawing it in place would have compressed the
# candles ~40%.
# --------------------------------------------------------------------------
def _marks(indexes, projected=None):
    return {"indexes": list(indexes), "projected": projected}


def _projection(**kwargs):
    from datetime import date

    base = {
        "date": date(2026, 10, 29),
        "cadence_days": 91,
        "sessions_ahead": 50,
        "overdue": False,
    }
    base.update(kwargs)
    return base


def test_an_e_is_drawn_for_every_earnings_bar():
    chart = _chart(_bars())
    try:
        chart.set_earnings(_marks([5, 20, 33]))
        assert chart.earnings_marker_count() == 3
    finally:
        chart.close()


def test_the_glyphs_ride_a_ribbon_above_every_candle():
    """A top ribbon, not a marker buried in the price action.

    Needs the reserved headroom to be true at all: these bars trend to the
    top-right, so without it the tallest candle reaches the ribbon line.
    """
    chart = _chart(_bars(), earnings=True)
    try:
        chart.set_earnings(_marks([5, 20]))
        _x, (y_min, y_max) = chart.getPlotItem().vb.viewRange()
        highs = [chart._y(bar["high"]) for bar in chart._bars]
        for item in chart._earnings_text_items:
            if not item.isVisible():
                continue
            assert item.pos().y() > max(highs), "an E must clear the candles"
            assert item.pos().y() < y_max, "and stay inside the view"
    finally:
        chart.close()


def test_each_e_sits_at_the_x_of_its_own_candle():
    chart = _chart(_bars())
    try:
        chart.set_earnings(_marks([5, 20, 33]))
        drawn = sorted(
            item.pos().x() for item in chart._earnings_text_items if item.isVisible()
        )
        assert drawn == [5.0, 20.0, 33.0]
    finally:
        chart.close()


def test_an_index_past_the_drawn_bars_is_ignored():
    """The payload is built against one bar list; a chart showing fewer must
    not index off the end."""
    chart = _chart(_bars(count=10))
    try:
        chart.set_earnings(_marks([3, 99, -1]))
        assert chart.earnings_marker_count() == 1
    finally:
        chart.close()


def test_the_projection_is_pinned_and_says_it_is_an_estimate():
    chart = _chart(_bars())
    try:
        chart.set_earnings(_marks([5], _projection()))
        text = chart.earnings_projection_text()
        assert "50d" in text
        assert "10/29" in text
        assert "est" in text.lower(), "a projected date must never read as known"
        # Pinned to the viewport, not to the data: it names a date beyond the
        # last bar and would otherwise pan off the chart.
        assert chart._earnings_projection.parentItem() is chart.getPlotItem().vb
    finally:
        chart.close()


def test_an_overdue_projection_says_due_rather_than_a_negative_countdown():
    chart = _chart(_bars())
    try:
        chart.set_earnings(_marks([5], _projection(overdue=True, sessions_ahead=0)))
        text = chart.earnings_projection_text()
        assert "due" in text.lower()
        assert "-" not in text
    finally:
        chart.close()


def test_no_earnings_payload_draws_nothing_and_never_raises():
    chart = _chart(_bars())
    try:
        chart.set_earnings(None)
        assert chart.earnings_marker_count() == 0
        assert chart.earnings_projection_text() == ""
        chart.set_earnings({})
        assert chart.earnings_marker_count() == 0
    finally:
        chart.close()


def test_the_axis_is_not_extended_to_reach_the_projection():
    """The trader's explicit choice: candles keep every pixel."""
    bars = _bars()
    plain = _chart(bars)
    with_projection = _chart(bars)
    try:
        with_projection.set_earnings(_marks([5], _projection()))
        assert (
            with_projection.getPlotItem().vb.viewRange()[0]
            == plain.getPlotItem().vb.viewRange()[0]
        )
    finally:
        plain.close()
        with_projection.close()


def test_the_ribbon_follows_a_pan_instead_of_staying_put():
    """The E belongs to a candle, so it moves with it."""
    chart = _chart(_bars())
    try:
        chart.set_earnings(_marks([5]))
        before = chart._earnings_text_items[0].pos().y()
        _x, (y_min, y_max) = chart.getPlotItem().vb.viewRange()
        span = y_max - y_min
        chart.getPlotItem().vb.setYRange(y_min + span, y_max + span, padding=0)
        assert chart._earnings_text_items[0].pos().y() > before
    finally:
        chart.close()


def test_the_d1_chart_receives_earnings_from_the_worker_payload():
    """Built on the worker beside the levels, never read on the paint path."""
    from ui.services.chart_data_service import ChartDataService

    marks = ChartDataService._build_earnings("AAPL", _bars())
    assert isinstance(marks, dict)
    assert "indexes" in marks and "projected" in marks


def test_a_broken_earnings_lookup_costs_the_markers_not_the_chart():
    from ui.services.chart_data_service import ChartDataService

    import chart_snapshot

    original = chart_snapshot.symbol_earnings_dates
    chart_snapshot.symbol_earnings_dates = lambda *_a, **_k: (_ for _ in ()).throw(
        OSError("cache gone")
    )
    try:
        assert ChartDataService._build_earnings("AAPL", _bars()) == {}
    finally:
        chart_snapshot.symbol_earnings_dates = original


def test_the_headroom_is_reserved_for_every_symbol_not_just_earnings_names():
    """Two symbols with identical prices must draw at the same scale."""
    bars = _bars()
    has_earnings = _chart(bars, earnings=True)
    no_earnings = _chart(bars, earnings=True)
    try:
        has_earnings.set_earnings(_marks([5], _projection()))
        no_earnings.set_earnings(_marks([]))
        assert (
            has_earnings.getPlotItem().vb.viewRange()[1]
            == no_earnings.getPlotItem().vb.viewRange()[1]
        )
    finally:
        has_earnings.close()
        no_earnings.close()


def test_a_chart_without_the_rail_keeps_its_old_range():
    """The M5 chart and every other CandleChart are untouched."""
    bars = _bars()
    plain = _chart(bars, earnings=False)
    railed = _chart(bars, earnings=True)
    try:
        assert (
            railed.getPlotItem().vb.viewRange()[1][1]
            > plain.getPlotItem().vb.viewRange()[1][1]
        )
    finally:
        plain.close()
        railed.close()


def test_the_d1_chart_asks_for_the_rail_and_the_m5_does_not():
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    widget = SymbolSnapshotWidget()
    try:
        assert widget.d1_chart._show_earnings is True
        assert widget.m5_chart._show_earnings is False
    finally:
        widget.deleteLater()
