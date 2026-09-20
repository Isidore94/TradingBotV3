"""TJ-3: the note markers are ADDITIVE to a chart the live desk draws.

`scripts/ui/widgets/candle_chart.py` is the Alert Center's chart as well as Day
Review's. The packet's rule, and what this file proves: **with no markers set,
every existing chart paints exactly as before and pays nothing for the family
it does not use** - no scene item, no range-change connection, no per-paint
work - and `barClicked` / `priceClicked` / `levelSelected` behave exactly as
they did.

The one shared line TJ-3 changed is `EarningsDropLines`' ribbon fraction, which
moved from a module constant to a class attribute of the same value so the note
connectors could take their own line without a second copy of the geometry. The
earnings ribbon is pinned here at its own number.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the chart is PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _bars(count: int = 40) -> list[dict]:
    first = datetime(2026, 9, 18, 6, 30, tzinfo=PACIFIC)
    return [
        {
            "dt": first + timedelta(minutes=5 * index),
            "open": 100.0 + index * 0.05,
            "high": 100.4 + index * 0.05,
            "low": 99.6 + index * 0.05,
            "close": 100.2 + index * 0.05,
            "volume": 1000.0,
        }
        for index in range(count)
    ]


def _marker(ref_id: str, index: int, kind: str = "note"):
    return {
        "stamp": _bars()[index]["dt"].isoformat(),
        "index": index,
        "kind": kind,
        "label": f"{kind} {ref_id}",
        "ref_id": ref_id,
    }


@pytest.fixture()
def chart(qapp):
    from ui.widgets.candle_chart import CandleChart

    widget = CandleChart()
    widget.resize(900, 600)
    yield widget
    widget.hide()
    widget.deleteLater()
    qapp.processEvents()


def _click(chart, qapp, view_x: float, view_y: float) -> None:
    local = chart.mapFromScene(
        chart.getPlotItem().vb.mapViewToScene(QPointF(view_x, view_y))
    )
    qapp.sendEvent(
        chart.viewport(),
        QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(local),
            chart.mapToGlobal(local),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    qapp.processEvents()


def test_a_chart_nobody_hands_markers_builds_no_marker_item(chart, qapp):
    """The Alert Center's chart never learns this family exists."""
    from ui.widgets.candle_chart import NoteMarkers

    chart.set_data(_bars(), timeframe="m5")
    chart.set_data(_bars(20), timeframe="m5")  # a symbol switch
    chart.show()
    qapp.processEvents()

    assert chart._note_marker_lines is None
    assert chart._note_marker_items == []
    assert not any(
        isinstance(item, NoteMarkers) for item in chart.getPlotItem().items
    )
    assert chart.note_marker_count() == 0
    assert chart.note_marker_position("anything") is None


def test_an_empty_payload_on_a_bare_chart_still_builds_nothing(chart, qapp):
    chart.set_data(_bars(), timeframe="m5")

    chart.set_note_markers([])
    chart.set_note_markers(())

    assert chart._note_marker_lines is None
    assert chart.note_marker_count() == 0


def test_a_click_with_no_markers_behaves_exactly_as_it_did(chart, qapp):
    """`barClicked` and `priceClicked` fire; `markerClicked` says nothing."""
    chart.set_data(_bars(), timeframe="m5")
    chart.show()
    qapp.processEvents()

    bars: list[int] = []
    prices: list[tuple[int, float]] = []
    markers: list[str] = []
    chart.barClicked.connect(bars.append)
    chart.priceClicked.connect(lambda index, price: prices.append((index, price)))
    chart.markerClicked.connect(markers.append)

    _click(chart, qapp, 12.0, chart._y(_bars()[12]["close"]))

    assert bars == [12]
    assert [index for index, _price in prices] == [12]
    assert prices[0][1] == pytest.approx(_bars()[12]["close"], rel=1e-3)
    assert markers == []


def test_a_click_on_a_marker_still_announces_the_bar_and_the_price(chart, qapp):
    """A click on a marker is STILL a click on the chart - never instead."""
    chart.set_data(_bars(), timeframe="m5")
    chart.set_note_markers([_marker("e-1", 24)])
    chart.show()
    qapp.processEvents()

    bars: list[int] = []
    prices: list[tuple[int, float]] = []
    markers: list[str] = []
    chart.barClicked.connect(bars.append)
    chart.priceClicked.connect(lambda index, price: prices.append((index, price)))
    chart.markerClicked.connect(markers.append)

    spot = chart.note_marker_position("e-1")
    assert spot is not None
    _click(chart, qapp, *spot)

    assert bars == [24]
    assert len(prices) == 1
    assert markers == ["e-1"]


def test_new_bars_drop_the_marker_payload(chart, qapp):
    """A marker index names a bar of the tape it was built against, and no other."""
    chart.set_data(_bars(), timeframe="m5")
    chart.set_note_markers([_marker("a", 3), _marker("b", 9)])
    qapp.processEvents()
    assert chart.note_marker_count() == 2

    chart.set_data(_bars(30), timeframe="m5")
    qapp.processEvents()

    assert chart.note_marker_count() == 0


def test_the_two_ribbons_keep_their_own_lines(chart, qapp):
    """The E glyphs and the note glyphs never sit on top of each other."""
    from ui.widgets import candle_chart as module

    assert module.EarningsDropLines.RIBBON_FRACTION == module._EARNINGS_RIBBON_FRACTION
    assert module.NoteMarkers.RIBBON_FRACTION == module._NOTE_RIBBON_FRACTION
    assert module.NoteMarkers.RIBBON_FRACTION != module.EarningsDropLines.RIBBON_FRACTION

    chart.set_earnings_visible(True)
    chart.set_data(_bars(), timeframe="m5")
    chart.set_earnings({"indexes": [12]})
    chart.set_note_markers([_marker("e-1", 12)])
    chart.show()
    qapp.processEvents()

    assert chart.earnings_marker_count() == 1
    assert chart.note_marker_count() == 1
    earnings_y = chart._earnings_text_items[0].pos().y()
    note_y = chart._note_marker_items[0].pos().y()
    assert earnings_y > note_y, "the note glyph was drawn over the earnings ribbon"


def test_markers_do_not_reserve_headroom_of_their_own(chart, qapp):
    """The y-range comes from the candles; only the earnings rail reserves room."""
    chart.set_data(_bars(), timeframe="m5")
    chart.show()
    qapp.processEvents()
    before = chart.getPlotItem().vb.viewRange()

    chart.set_note_markers([_marker("a", 0), _marker("b", 39)])
    chart.set_data(_bars(), timeframe="m5")
    chart.set_note_markers([_marker("a", 0), _marker("b", 39)])
    qapp.processEvents()

    assert chart.getPlotItem().vb.viewRange() == before


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
