"""A4 paint-lines, widget side: rendering, the toggle, and click-to-select.

The data half (level building, ids, the trendline projection, the off-thread
guarantee) lives in ``test_chart_levels.py``.
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

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

import chart_levels  # noqa: E402
from ui.widgets.candle_chart import LEVEL_HIT_TOLERANCE_PX, CandleChart  # noqa: E402


def _bars(count: int = 30, base: float = 100.0) -> list[dict]:
    first = datetime(2026, 6, 1)
    out = []
    for index in range(count):
        close = base + index * 0.1
        out.append(
            {
                "dt": first + timedelta(days=index),
                "open": close - 0.2,
                "high": close + 0.4,
                "low": close - 0.4,
                "close": close,
                "volume": 1_000.0,
            }
        )
    return out


def _level(level_id: str, price: float, group: str | None = None, **extra) -> dict:
    level = {
        "id": level_id,
        "family": "d1_horizontal",
        "group": group or chart_levels.GROUP_HORIZONTAL,
        "price": price,
        "values": None,
        "label": f"L {price}",
        "color": "chart_green",
        "width": 1.2,
        "dash": False,
        "conviction": 1.0,
    }
    level.update(extra)
    return level


def _chart(bars=None, levels=()) -> CandleChart:
    chart = CandleChart()
    chart.resize(600, 400)
    bars = _bars() if bars is None else bars
    chart.set_data(bars, [], timeframe="d1")
    chart.set_levels(list(levels))
    return chart


# --------------------------------------------------------------------------
# rendering + the toggle
# --------------------------------------------------------------------------
def test_horizontal_levels_render_and_hide():
    chart = _chart(levels=[_level("a", 101.0), _level("b", 102.0)])
    assert [level["id"] for level in chart.drawn_levels()] == ["a", "b"]
    visible = [item for item in chart._level_line_items if item.isVisible()]
    assert len(visible) == 2

    # The toggle's effect: the host filters and re-pushes.
    chart.set_levels(
        chart_levels.visible_levels(
            [_level("a", 101.0), _level("b", 102.0)],
            [chart_levels.GROUP_HORIZONTAL],
        )
    )
    assert chart.drawn_levels() == []
    assert not any(item.isVisible() for item in chart._level_line_items)


def test_hidden_levels_reuse_their_pooled_items():
    """Same C5 discipline as the overlays: hide, never destroy."""
    chart = _chart(levels=[_level("a", 101.0), _level("b", 102.0)])
    identities = [id(item) for item in chart._level_line_items]
    chart.set_levels([_level("a", 101.0)])
    chart.set_levels([_level("a", 101.0), _level("b", 102.0)])
    assert [id(item) for item in chart._level_line_items] == identities


def test_a_sloped_level_renders_as_a_curve():
    bars = _bars()
    values = [100.0 + index * 0.05 for index in range(len(bars))]
    line = _level(
        "tl",
        values[-1],
        chart_levels.GROUP_TRENDLINE,
        values=values,
        family="d1_trendline",
    )
    chart = _chart(bars, [line])
    assert [level["id"] for level in chart.drawn_levels()] == ["tl"]
    assert sum(item.isVisible() for item in chart._level_curve_items) == 1
    assert not any(item.isVisible() for item in chart._level_line_items)


def test_a_series_that_does_not_align_with_the_bars_is_not_drawn():
    bars = _bars(10)
    line = _level("tl", 100.0, chart_levels.GROUP_TRENDLINE, values=[1.0, 2.0])
    chart = _chart(bars, [line])
    assert chart.drawn_levels() == []


def test_levels_clear_when_the_chart_has_no_bars():
    chart = _chart(levels=[_level("a", 101.0)])
    chart.set_data([], [])
    assert chart.drawn_levels() == []


# --------------------------------------------------------------------------
# the y-range guarantee
# --------------------------------------------------------------------------
def test_an_off_screen_level_never_expands_the_y_range():
    """The chart's y-range follows the candles. A level does not get a vote."""
    bars = _bars()
    chart = _chart(bars, [])
    before = chart.getPlotItem().vb.viewRange()[1]
    chart.set_levels([_level("far", 10_000.0), _level("low", 0.01)])
    after = chart.getPlotItem().vb.viewRange()[1]
    assert after == pytest.approx(before)
    # And the same on a re-render, where set_data re-ranges from the candles.
    chart.set_data(bars, [], timeframe="d1")
    assert chart.getPlotItem().vb.viewRange()[1] == pytest.approx(before)


def test_an_off_screen_sloped_level_never_expands_the_y_range():
    bars = _bars()
    chart = _chart(bars, [])
    before = chart.getPlotItem().vb.viewRange()[1]
    chart.set_levels(
        [
            _level(
                "tl",
                9_000.0,
                chart_levels.GROUP_TRENDLINE,
                values=[9_000.0] * len(bars),
            )
        ]
    )
    assert chart.getPlotItem().vb.viewRange()[1] == pytest.approx(before)


# --------------------------------------------------------------------------
# click hit-testing
# --------------------------------------------------------------------------
def _pixel_height(chart: CandleChart) -> float:
    return float(chart.getPlotItem().vb.viewPixelSize()[1])


def test_a_click_within_tolerance_selects_the_level():
    bars = _bars()
    price = bars[10]["close"]
    chart = _chart(bars, [_level("hit", price)])
    view_y = chart._y(price) + _pixel_height(chart) * (LEVEL_HIT_TOLERANCE_PX - 2)
    hit = chart.level_at(10, view_y)
    assert hit is not None and hit["id"] == "hit"


def test_a_click_outside_tolerance_misses():
    bars = _bars()
    price = bars[10]["close"]
    chart = _chart(bars, [_level("miss", price)])
    view_y = chart._y(price) + _pixel_height(chart) * (LEVEL_HIT_TOLERANCE_PX + 4)
    assert chart.level_at(10, view_y) is None


def test_the_nearest_level_wins_when_two_are_close():
    bars = _bars()
    step = _pixel_height(chart := _chart(bars, []))
    base = bars[10]["close"]
    near = chart.price_at(chart._y(base) + step * 1.0)
    far = chart.price_at(chart._y(base) + step * 4.0)
    chart.set_levels([_level("near", near), _level("far", far)])
    hit = chart.level_at(10, chart._y(base) + step * 1.2)
    assert hit["id"] == "near"


def test_hit_testing_a_sloped_level_uses_the_value_at_that_bar():
    bars = _bars()
    values = [100.0 + index for index in range(len(bars))]
    line = _level("tl", values[-1], chart_levels.GROUP_TRENDLINE, values=values)
    chart = _chart(bars, [line])
    # At bar 3 the line sits at 103, nowhere near its last value.
    assert chart.level_at(3, chart._y(103.0)) is not None
    assert chart.level_at(3, chart._y(values[-1])) is None


def test_hit_testing_skips_a_bar_where_the_sloped_level_is_undefined():
    bars = _bars(10)
    values = [None] * 5 + [100.0 + index for index in range(5, 10)]
    line = _level("tl", values[-1], chart_levels.GROUP_TRENDLINE, values=values)
    chart = _chart(bars, [line])
    assert chart.level_at(2, chart._y(102.0)) is None
    assert chart.level_at(7, chart._y(values[7])) is not None


def test_selection_emits_the_levels_identity_and_highlights_it():
    bars = _bars()
    price = bars[10]["close"]
    chart = _chart(bars, [_level("hit", price, family="d1_horizontal")])
    seen: list[tuple] = []
    chart.levelSelected.connect(lambda *args: seen.append(args))
    chart._select_level_at(10, chart._y(price))
    assert seen == [("hit", "d1_horizontal", pytest.approx(price))]
    assert chart.selected_level_id() == "hit"


def test_clicking_away_clears_the_selection():
    bars = _bars()
    price = bars[10]["close"]
    chart = _chart(bars, [_level("hit", price)])
    chart._select_level_at(10, chart._y(price))
    assert chart.selected_level_id() == "hit"
    chart._select_level_at(10, chart._y(price) + _pixel_height(chart) * 40)
    assert chart.selected_level_id() == ""


def test_a_selection_that_leaves_the_chart_is_dropped():
    bars = _bars()
    price = bars[10]["close"]
    chart = _chart(bars, [_level("hit", price)])
    chart.select_level("hit")
    chart.set_levels([_level("other", price + 5)])
    assert chart.selected_level_id() == ""


def test_selecting_thickens_the_line_without_recoloring_it():
    bars = _bars()
    price = bars[10]["close"]
    level = _level("hit", price)
    chart = _chart(bars, [level])
    plain = chart._level_pen(level, False)
    highlighted = chart._level_pen(level, True)
    assert highlighted.widthF() > plain.widthF()
    assert highlighted.color() == plain.color()


# --------------------------------------------------------------------------
# preferences
# --------------------------------------------------------------------------
def test_paint_lines_prefs_default_to_everything_on(tmp_path):
    from ui.services.paint_lines_prefs import PaintLinesPrefs

    prefs = PaintLinesPrefs(tmp_path / "paint.json")
    assert prefs.hidden_groups() == []
    assert all(prefs.is_visible(group) for group, _label in chart_levels.LEVEL_GROUPS)


def test_paint_lines_prefs_round_trip_and_survive_a_reload(tmp_path):
    from ui.services.paint_lines_prefs import PaintLinesPrefs

    path = tmp_path / "paint.json"
    prefs = PaintLinesPrefs(path)
    prefs.set_visible(chart_levels.GROUP_HORIZONTAL, False)
    assert PaintLinesPrefs(path).hidden_groups() == [chart_levels.GROUP_HORIZONTAL]
    prefs.set_visible(chart_levels.GROUP_HORIZONTAL, True)
    assert PaintLinesPrefs(path).hidden_groups() == []


def test_paint_lines_prefs_ignore_junk_and_unknown_groups(tmp_path):
    from ui.services.paint_lines_prefs import PaintLinesPrefs

    path = tmp_path / "paint.json"
    path.write_text('{"hidden_groups": ["nope", "sma"]}', encoding="utf-8")
    assert PaintLinesPrefs(path).hidden_groups() == ["sma"]
    path.write_text("not json", encoding="utf-8")
    assert PaintLinesPrefs(path).hidden_groups() == []


def test_paint_lines_prefs_are_machine_local(tmp_path):
    """A display preference must not ride Drive to another machine."""
    from project_paths import LOCAL_SETTINGS_DIR, PERSISTENT_DATA_DIR
    from ui.services.paint_lines_prefs import PAINT_LINES_FILE

    assert LOCAL_SETTINGS_DIR in PAINT_LINES_FILE.parents
    assert PERSISTENT_DATA_DIR not in PAINT_LINES_FILE.parents


def test_paint_lines_button_reports_and_persists_the_hidden_set(tmp_path):
    from ui.services.paint_lines_prefs import PaintLinesPrefs
    from ui.widgets.paint_lines_button import PaintLinesButton

    path = tmp_path / "paint.json"
    button = PaintLinesButton(prefs=PaintLinesPrefs(path))
    emitted: list[list] = []
    button.groupsChanged.connect(lambda groups: emitted.append(list(groups)))
    assert button.hidden_groups() == []
    assert "off" not in button.text()

    action = button._actions[chart_levels.GROUP_HORIZONTAL]
    action.setChecked(False)
    assert emitted[-1] == [chart_levels.GROUP_HORIZONTAL]
    assert button.hidden_groups() == [chart_levels.GROUP_HORIZONTAL]
    assert "1 off" in button.text()
    assert PaintLinesPrefs(path).hidden_groups() == [chart_levels.GROUP_HORIZONTAL]


@pytest.mark.parametrize("compact_density", [False, True])
@pytest.mark.parametrize("width", [2560, 1280])
def test_the_compact_lines_button_costs_the_header_row_no_height(
    tmp_path, compact_density, width
):
    """The whole reason the compact button is flat: pixels for the candles.

    Measured under the real stylesheet, because the default style has no
    button padding and would hide the very chrome that used to cost the
    embedded pane 5px of chart.
    """
    from ui import theme
    from ui.services.paint_lines_prefs import PaintLinesPrefs
    from ui.widgets.paint_lines_button import PaintLinesButton
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    def _measure(with_button: bool) -> tuple[tuple[int, int], int]:
        widget = SymbolSnapshotWidget(compact=True)
        widget.paint_lines_button.setParent(None)
        if with_button:
            widget.paint_lines_button = PaintLinesButton(
                compact=True, prefs=PaintLinesPrefs(tmp_path / "p.json")
            )
            widget.d1_header.layout().addWidget(widget.paint_lines_button, 0)
        widget.resize(width, 1200)
        widget.show()
        _app.processEvents()
        sizes = (widget.d1_header.height(), widget.d1_chart.height())
        button = widget.paint_lines_button
        # How tall the button wants to be, above one line of its own text.
        chrome = button.minimumSizeHint().height() - button.fontMetrics().height()
        widget.hide()
        widget.deleteLater()
        return sizes, chrome

    previous_sheet = _app.styleSheet()
    try:
        theme.apply_theme(_app, "dark", compact_density, 1.0)
        with_button, chrome = _measure(True)
        without_button, _ = _measure(False)
    finally:
        _app.setStyleSheet(previous_sheet)
        _app.processEvents()

    assert with_button == without_button
    # Capping the height alone is not enough. A button still wearing the
    # theme's button padding and border gets squeezed into the row and paints
    # a sliver of "Lines" instead of the word, so the chrome it asks for must
    # stay inside the style's own few pixels of trim.
    assert chrome <= 6


# --------------------------------------------------------------------------
# the host: one snapshot, filtered on the way to the chart
# --------------------------------------------------------------------------
def _snapshot(bars: list[dict]) -> dict:
    return {
        "symbol": "AAA",
        "timeframe": "D1",
        "bars": bars,
        "overlays": [
            {
                "label": "SMA50",
                "values": [b["close"] for b in bars],
                "color": "chart_light_blue",
                "width": 1.6,
                "dash": "dot",
            },
            {
                "label": "EMA21",
                "values": [b["close"] for b in bars],
                "color": "chart_yellow",
                "width": 1.1,
                "dash": False,
            },
        ],
        "levels": [
            _level("sr", bars[5]["close"]),
            _level("pdh", bars[-2]["high"], chart_levels.GROUP_PREV_DAY,
                   family="prev_day_high"),
        ],
        "note": "",
    }


def test_the_snapshot_widget_paints_levels_and_the_toggle_hides_them(tmp_path):
    from ui.services.paint_lines_prefs import PaintLinesPrefs
    from ui.widgets.paint_lines_button import PaintLinesButton
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    widget = SymbolSnapshotWidget()
    widget.paint_lines_button.setParent(None)
    widget.paint_lines_button = PaintLinesButton(prefs=PaintLinesPrefs(tmp_path / "p.json"))
    widget.paint_lines_button.groupsChanged.connect(widget._on_paint_lines_changed)
    widget._symbol = "AAA"

    bars = _bars()
    widget._render_snapshots(_snapshot(bars), {"bars": [], "overlays": [], "note": ""})
    assert {level["id"] for level in widget.d1_chart.drawn_levels()} == {"sr", "pdh"}

    widget.paint_lines_button._actions[chart_levels.GROUP_PREV_DAY].setChecked(False)
    assert {level["id"] for level in widget.d1_chart.drawn_levels()} == {"sr"}

    widget.paint_lines_button._actions[chart_levels.GROUP_PREV_DAY].setChecked(True)
    assert {level["id"] for level in widget.d1_chart.drawn_levels()} == {"sr", "pdh"}
    widget.deleteLater()


def test_the_toggle_hides_the_overlays_it_names_too():
    """The SMAs and AVWAP bands are already painted; A4 owns their switch."""
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    widget = SymbolSnapshotWidget()
    widget._symbol = "AAA"
    bars = _bars()
    snapshot = _snapshot(bars)
    widget._render_snapshots(snapshot, {"bars": [], "overlays": [], "note": ""})
    assert len(widget.d1_chart._overlays) == 2

    widget.paint_lines_button._prefs.set_visible(chart_levels.GROUP_SMA, False)
    widget._on_paint_lines_changed([chart_levels.GROUP_SMA])
    assert [o["label"] for o in widget.d1_chart._overlays] == ["EMA21"]
    widget.paint_lines_button._prefs.set_visible(chart_levels.GROUP_SMA, True)
    widget.deleteLater()


def test_toggling_lines_does_not_re_range_the_chart():
    """A display switch must not throw away the trader's pan and zoom."""
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    widget = SymbolSnapshotWidget()
    widget._symbol = "AAA"
    bars = _bars()
    widget._render_snapshots(_snapshot(bars), {"bars": [], "overlays": [], "note": ""})
    widget.d1_chart.getPlotItem().setXRange(5, 15, padding=0)

    def _range() -> list[float]:
        x, y = widget.d1_chart.getPlotItem().vb.viewRange()
        return [*x, *y]

    before = _range()
    widget.paint_lines_button._prefs.set_visible(chart_levels.GROUP_PREV_DAY, False)
    widget._on_paint_lines_changed([chart_levels.GROUP_PREV_DAY])
    assert _range() == pytest.approx(before)
    widget.paint_lines_button._prefs.set_visible(chart_levels.GROUP_PREV_DAY, True)
    widget.deleteLater()


def test_the_widget_forwards_a_level_click_with_its_symbol():
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    widget = SymbolSnapshotWidget()
    widget._symbol = "AAA"
    bars = _bars()
    widget._render_snapshots(_snapshot(bars), {"bars": [], "overlays": [], "note": ""})
    seen: list[tuple] = []
    widget.d1LevelSelected.connect(lambda *args: seen.append(args))

    price = bars[5]["close"]
    widget.d1_chart._select_level_at(5, widget.d1_chart._y(price))
    assert seen and seen[-1][0] == "AAA" and seen[-1][1] == "sr"
    chosen = widget.selected_d1_level()
    assert chosen is not None and chosen["id"] == "sr"
    widget.deleteLater()
