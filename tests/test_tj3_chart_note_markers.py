"""TJ-3 item 1 - the pooled `NoteMarkers` overlay on the chart. RED first.

Packet `.claude/packets/TJ-3.md`; `plan.md` §12.4 "TJ-3" change 1.

The contract these tests pin, so the builder has nothing to guess
-----------------------------------------------------------------

``scripts/ui/widgets/candle_chart.py`` gains ONE overlay family and nothing
else - the same pooling discipline `EarningsDropLines` already keeps (one
`pg.TextItem` per marker, hidden rather than destroyed, never rebuilt):

``NoteMarkers``
    the connector geometry, one item, like `EarningsDropLines`.
``CandleChart.set_note_markers(markers)``
    the payload `day_review_markers` built on the worker. Draws; reads nothing.
``CandleChart.note_marker_count() -> int``
    how many markers are VISIBLE right now (the `earnings_marker_count`
    precedent).
``CandleChart.note_marker_position(ref_id) -> tuple[float, float] | None``
    where that marker's glyph sits, in the view's own coordinates - what a test
    (and a future hit test) needs to find it without guessing the ribbon.
``CandleChart.markerClicked = Signal(str)``
    the marker's ``ref_id``, emitted when a real left click lands on it. In
    ADDITION to ``barClicked``/``priceClicked``, never instead: a click on a
    marker is still a click on the chart.

The repo's overlay rule holds for this family too: the price scale comes from
the candles and nothing else, so markers may not move the y-range.
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
    widget.set_data(_bars(), timeframe="m5")
    yield widget
    widget.hide()
    widget.deleteLater()
    qapp.processEvents()


def _scene_items(chart):
    return list(chart.getPlotItem().items)


def test_the_marker_glyphs_pool_across_three_renders(chart, qapp):
    """Three renders of the same payload build ONE set of items and reuse it.

    The `EarningsDropLines` discipline: text has to be a scene item, and a
    chart that constructs scene items per render is the cost C5 removed from
    the paint path.
    """
    payload = [_marker("a", 5), _marker("b", 17, "veto"), _marker("c", 31, "trade_open")]

    # The ITEMS are held, not their `id()`: a freed scene item's address can be
    # handed straight back to its replacement, and then a rebuild would read as
    # a reuse.
    seen = []
    for _pass in range(3):
        chart.set_note_markers(payload)
        qapp.processEvents()
        seen.append(_scene_items(chart))

    assert chart.note_marker_count() == 3
    assert len(seen[0]) == len(seen[1]) == len(seen[2])
    for first, second, third in zip(*seen):
        assert first is second is third, "the glyphs were rebuilt instead of reused"


def test_a_shorter_payload_hides_the_spare_glyphs_and_destroys_nothing(chart, qapp):
    chart.set_note_markers([_marker(name, index) for name, index in
                            (("a", 3), ("b", 9), ("c", 15), ("d", 21))])
    qapp.processEvents()
    assert chart.note_marker_count() == 4
    full = _scene_items(chart)

    chart.set_note_markers([_marker("a", 3), _marker("b", 9)])
    qapp.processEvents()

    assert chart.note_marker_count() == 2
    after = _scene_items(chart)
    assert len(after) == len(full)
    for kept, now in zip(full, after):
        assert kept is now, "a spare glyph was destroyed instead of hidden"


def test_no_markers_hides_the_family_without_touching_the_candles(chart, qapp):
    chart.set_note_markers([_marker("a", 3)])
    qapp.processEvents()

    chart.set_note_markers([])
    qapp.processEvents()

    assert chart.note_marker_count() == 0
    assert chart.bar_count() == 40


def test_a_marker_with_no_index_is_not_drawn(chart, qapp):
    """An unplaceable marker is dropped, never parked at bar zero."""
    unplaced = dict(_marker("ghost", 0))
    unplaced["index"] = None

    chart.set_note_markers([_marker("real", 12), unplaced])
    qapp.processEvents()

    assert chart.note_marker_count() == 1
    assert chart.note_marker_position("ghost") is None


def test_a_marker_off_the_end_of_the_bars_is_not_drawn(chart, qapp):
    """A shorter tape than the payload was built against hides the marker."""
    chart.set_note_markers([_marker("real", 12), _marker("beyond", 39) | {"index": 400}])
    qapp.processEvents()

    assert chart.note_marker_count() == 1


def test_markers_never_move_the_price_range(chart, qapp):
    """The scale comes from the candles; no overlay gets a vote in it."""
    chart.show()
    qapp.processEvents()
    before = chart.getPlotItem().vb.viewRange()

    chart.set_note_markers([_marker("a", 2), _marker("b", 37)])
    qapp.processEvents()

    assert chart.getPlotItem().vb.viewRange() == before


def test_a_real_click_on_a_marker_emits_its_ref_id(chart, qapp):
    """Delivered the way Qt delivers it - to the viewport, not to a helper."""
    chart.set_note_markers([_marker("e-first", 8), _marker("e-second", 24)])
    chart.show()
    qapp.processEvents()

    seen: list[str] = []
    chart.markerClicked.connect(seen.append)

    spot = chart.note_marker_position("e-second")
    assert spot is not None, "the chart cannot say where it drew the marker"
    local = chart.mapFromScene(chart.getPlotItem().vb.mapViewToScene(QPointF(*spot)))
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

    assert seen == ["e-second"]


def test_a_click_on_a_bare_candle_emits_no_marker(chart, qapp):
    chart.set_note_markers([_marker("e-first", 8)])
    chart.show()
    qapp.processEvents()

    seen: list[str] = []
    chart.markerClicked.connect(seen.append)

    # The candle's own close, thirty bars away from the only marker.
    local = chart.mapFromScene(
        chart.getPlotItem().vb.mapViewToScene(
            QPointF(38.0, chart._y(_bars()[38]["close"]))
        )
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

    assert seen == []


def test_new_bars_do_not_leave_the_old_markers_on_the_chart(chart, qapp):
    """A symbol switch is new bars; a stale marker would name the wrong candle."""
    chart.set_note_markers([_marker("a", 3), _marker("b", 9)])
    qapp.processEvents()

    chart.set_data(_bars(12), timeframe="m5")
    chart.set_note_markers([_marker("a", 3)])
    qapp.processEvents()

    assert chart.note_marker_count() == 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
