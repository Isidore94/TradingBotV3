"""TJ-1L - the Day Review page in TWO COLUMNS, with tables that fill the width.

Trader, 2026-09-18, after the first look at the page on a 3800 px wide screen:
*"there's a lot of empty space horizontally that's not being efficiently used"* -
and, offered three shapes, chose **two columns**.

This file is presentation only and it pins the SHAPE, not a pixel:

* Row 1 is the session picker, full width, as it already was.
* Row 2 is a horizontal ``QSplitter`` named ``DayReviewColumns``, default 55/45,
  restored from and saved to the same per-machine settings file every other
  desk splitter uses. LEFT: *What happened*, *Open theses* directly under it,
  then the one lazily built SPY chart (a chart wants width). RIGHT: the entries
  list over the reader, then *New entry*, its verb row, and the collapsed
  *External forecast*.
* Row 3 is *Walk-away* as a 2 x 2 grid of its four TJ-2B tables.
* Row 4 is *What you traded* beside *Ideas from the desk's AI*.
* Every table fills its cell: the last section stretches and the rest measure
  their contents, so "Against me first %" is no longer "ainst me first".

What it must NOT change: TJ-1's contract. ONE ``CandleChart``, built on first
need and reused, is re-asserted here because a layout rewrite is exactly the
change that could quietly build it twice.
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

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QTableWidget,
)

SESSION = "2026-09-10"
NOW = datetime(2026, 9, 11, 7, 30)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _StubService:
    """Reads nothing, writes nothing. The layout is what is under test."""

    def __init__(self, payload=None) -> None:
        self.payload = payload or {}

    def read_day(self, session_date, **_kwargs):
        payload = dict(self.payload)
        payload.setdefault("session_date", session_date)
        return payload


def _panel(qapp, **kwargs):
    from ui.panels.day_review_panel import DayReviewPanel

    widget = DayReviewPanel(service=_StubService(), clock=lambda: NOW, **kwargs)
    return widget


@pytest.fixture()
def panel(qapp, monkeypatch):
    widget = _panel(qapp)
    # `reload` is the thread starter; every test here paints through `render`.
    monkeypatch.setattr(widget, "reload", lambda: None)
    yield widget
    try:
        widget.shutdown()
    except Exception:
        pass
    widget.deleteLater()
    qapp.processEvents()


@pytest.fixture()
def clean_split_settings():
    """Put the two split keys back exactly as they were found.

    `conftest.py` has already pointed `LOCALAPPDATA` at an empty temp dir, so
    these writes never touch the trader's own `local_settings.json`; restoring
    them still matters because a stored split would leak into the next test.
    """
    from project_paths import get_local_setting, save_local_setting
    from ui.panels.day_review_panel import COLUMN_SPLIT_KEY, SAID_SPLIT_KEY

    keys = (COLUMN_SPLIT_KEY, SAID_SPLIT_KEY)
    before = {key: get_local_setting(key, None) for key in keys}
    yield
    for key, value in before.items():
        save_local_setting(key, value)


def _payload(**overrides):
    base = {
        "session_date": SESSION,
        "provisional": False,
        "story": None,
        "theses": [],
        "entries": [],
        "rejected_that_worked": [],
        "walkaway": None,
        "trades": [],
        "forecast": {},
        "spy_m5_bars": [],
    }
    base.update(overrides)
    return base


def _walkaway_day(rows):
    """A real `WalkawayDay` whose passed-and-ran population is `rows`' names."""
    from walkaway_day import WalkawayDay, WalkawayRow

    return WalkawayDay(rejected=tuple(
        WalkawayRow(
            decision_id=(SESSION, row.symbol, "LONG", "chart_review", "veto", "D1", ""),
            time=row.observed_at, symbol=row.symbol, side="LONG",
            category="chart_review", what_you_did="veto", ran_after_pct=1.25,
            held_at_close_pct=0.9, state="measured",
        )
        for row in rows
    ))


class _Row:
    """One `rejected_that_worked` row, shaped like the reader's own."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.side = "long"
        self.observed_at = datetime(2026, 9, 10, 7, 5)
        self.measures = {
            "favorable_pct": 1.25,
            "adverse_pct": -0.4,
            "favorable_pct_after_decision": 0.9,
        }
        self.unavailable: dict[str, str] = {}
        self.detail = {
            "verdict": "veto",
            "channel": "d1_scan",
            "reason": "extended into the move",
            "reason_codes": ["extended"],
        }
        self.observation_context = "trend up"
        self.entry_context = "pullback"
        self.d1_environment = "trend up"


SPY_BARS = [
    {
        "dt": datetime(2026, 9, 10, 6, 30) + timedelta(minutes=5 * i),
        "open": 100.0 + i,
        "high": 101.0 + i,
        "low": 99.5 + i,
        "close": 100.5 + i,
        "volume": 1000 + i,
    }
    for i in range(12)
]


def _is_descendant(widget, ancestor) -> bool:
    node = widget
    while node is not None:
        if node is ancestor:
            return True
        node = node.parentWidget()
    return False


def _labels(widget) -> list[str]:
    return [label.text() for label in widget.findChildren(QLabel)]


# ==========================================================================
# 1. two columns
# ==========================================================================
def test_the_page_holds_one_horizontal_splitter_named_for_the_two_columns(panel):
    from ui.panels.day_review_panel import COLUMNS_OBJECT_NAME

    splitters = [child for child in panel.findChildren(QSplitter) if child.orientation() == Qt.Orientation.Horizontal]
    assert len(splitters) == 1, f"one horizontal splitter, got {len(splitters)}"
    columns = splitters[0]
    assert columns is panel.columns
    assert columns.objectName() == COLUMNS_OBJECT_NAME == "DayReviewColumns"
    assert columns.count() == 2, "two columns, never three"
    # A column dragged to nothing is a column the trader cannot find again -
    # the same rule the desk's D1 column split is built to.
    assert columns.childrenCollapsible() is False


def test_the_default_split_is_fifty_five_forty_five(panel, qapp):
    from ui.panels.day_review_panel import COLUMN_WEIGHTS

    assert COLUMN_WEIGHTS == (55, 45)
    panel.resize(1900, 1000)
    panel.show()
    qapp.processEvents()
    sizes = panel.columns.sizes()
    try:
        share = sizes[0] / max(1, sum(sizes))
        assert 0.52 <= share <= 0.58, sizes
    finally:
        panel.hide()


def test_what_happened_the_theses_and_the_spy_chart_are_the_left_column(panel):
    left = panel.columns.widget(0)
    right = panel.columns.widget(1)
    for widget in (panel.story_note, panel.story_facts, panel.theses, panel.spy_note, panel._chart_holder):
        assert _is_descendant(widget, left), widget
        assert not _is_descendant(widget, right), widget


def test_the_theses_sit_directly_under_the_story_and_the_chart_last(panel):
    """ "Open theses directly UNDER it (not beside it)" - the trader's option 1."""
    layout = panel.left_column.layout()
    story = layout.indexOf(panel.story_section)
    theses = layout.indexOf(panel.theses_section)
    spy = layout.indexOf(panel.spy_section)
    assert story >= 0 and theses >= 0 and spy >= 0
    assert story < theses < spy, (story, theses, spy)


def test_what_you_said_the_new_entry_box_and_the_forecast_are_the_right_column(panel):
    left = panel.columns.widget(0)
    right = panel.columns.widget(1)
    for widget in (
        panel.entries,
        panel.entry_reader,
        panel.entry_text,
        panel.timeframe_picker,
        panel.save_button,
        panel.paste_forecast_button,
        panel.forecast_box,
        panel.forecast_toggle,
    ):
        assert _is_descendant(widget, right), widget
        assert not _is_descendant(widget, left), widget


def test_every_box_in_the_right_column_spans_the_column(panel):
    """The G3 100-character cap is what put the empty space back.

    Measured at 3800x2000 on the first build: the reader and the External
    forecast stopped at about 420 px of an 890 px column while `New entry`
    under them ran its full width. In a column the trader can drag, the COLUMN
    is the measure - the splitter is the control, not a cap he cannot see.
    """
    from PySide6.QtWidgets import QSizePolicy

    for widget in (panel.entry_text, panel.entry_reader, panel.forecast_box):
        assert widget.maximumWidth() >= 16_777_215 - 1, widget
        assert widget.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding, widget


def test_the_scale_seam_leaves_both_boxes_spanning(panel):
    """`MainWindow._apply_scaled_metrics` calls this on a scale change; it must
    not re-cap what it was written to cap."""
    panel.refresh_reader_measure()
    for widget in (panel.entry_reader, panel.forecast_box):
        assert widget.maximumWidth() >= 16_777_215 - 1, widget


def test_the_spy_chart_pane_has_room_to_be_a_chart(panel):
    from ui import theme
    from ui.panels.day_review_panel import SPY_MIN_HEIGHT_PX

    assert SPY_MIN_HEIGHT_PX == 320
    assert panel._chart_holder.minimumHeight() >= theme.px(SPY_MIN_HEIGHT_PX)


def test_the_story_box_starts_at_a_readable_height_and_grows_with_its_text(panel):
    from ui import theme
    from ui.panels.day_review_panel import STORY_MIN_HEIGHT_PX

    assert STORY_MIN_HEIGHT_PX == 120
    assert panel.story_facts.minimumHeight() >= theme.px(STORY_MIN_HEIGHT_PX)
    short = panel.story_section.sizeHint().height()
    panel.story_facts.setText("\n".join(f"line {i}" for i in range(30)))
    assert panel.story_section.sizeHint().height() >= short


# ==========================================================================
# 2. the entries list and the reader share a splitter
# ==========================================================================
def test_the_entries_list_and_the_reader_share_a_vertical_splitter(panel, qapp):
    from ui.panels.day_review_panel import SAID_SPLIT_WEIGHTS

    assert SAID_SPLIT_WEIGHTS == (60, 40)
    split = panel.said_split
    assert isinstance(split, QSplitter)
    assert split.orientation() == Qt.Orientation.Vertical
    assert split.count() == 2
    assert _is_descendant(panel.entries, split.widget(0))
    assert _is_descendant(panel.entry_reader, split.widget(1))

    panel.resize(1900, 1000)
    panel.show()
    qapp.processEvents()
    try:
        sizes = split.sizes()
        share = sizes[0] / max(1, sum(sizes))
        assert 0.55 <= share <= 0.65, sizes
    finally:
        panel.hide()


# ==========================================================================
# 3. the walk-away populations: ONE table, five chips (Day Recap step A)
# ==========================================================================
def test_the_walkaway_is_one_table_with_five_filter_chips(panel):
    """Day Recap step A (trader, 2026-09-23) replaced the 2 x 2 grid plus a
    full-width fifth row with ONE table and a chip per population."""
    from ui.panels.day_review_panel import MISS_FILTERS

    assert tuple(panel.miss_chips) == tuple(name for name, _label in MISS_FILTERS) == (
        "rejected", "liked_not_traded", "traded_left_early", "claimed_d1", "earlier_calls",
    )
    assert _is_descendant(panel.miss_table, panel.miss_section)
    for chip in panel.miss_chips.values():
        assert _is_descendant(chip, panel.miss_section)
    assert panel.miss_population() == "rejected"


def test_the_name_chart_sits_beside_the_table_and_takes_the_rest(panel, qapp):
    panel.resize(1900, 1000)
    panel.show()
    qapp.processEvents()
    try:
        table = panel.miss_table.geometry()
        chart = panel._name_chart_holder
        assert chart.mapTo(panel, chart.rect().topLeft()).x() > panel.miss_table.mapTo(
            panel, panel.miss_table.rect().topLeft()
        ).x() + table.width() - 1
        assert chart.width() >= table.width() / 2, (chart.width(), table.width())
    finally:
        panel.hide()


def test_the_one_table_keeps_every_column_behind_more_columns(panel):
    from ui.panels.day_review_panel import MISS_DEFAULT_COLUMNS, TJ2B_WALKAWAY_COLUMNS

    table = panel.miss_table
    assert table.columnCount() == len(TJ2B_WALKAWAY_COLUMNS)
    shown = [
        TJ2B_WALKAWAY_COLUMNS[index]
        for index in range(table.columnCount()) if not table.isColumnHidden(index)
    ]
    assert tuple(shown) == MISS_DEFAULT_COLUMNS
    panel.more_columns_toggle.setChecked(True)
    assert not any(table.isColumnHidden(index) for index in range(table.columnCount()))

# ==========================================================================
# 4. what you traded, beside the ideas
# ==========================================================================
def test_what_you_traded_and_the_ideas_card_sit_in_one_row(panel):
    row = panel.bottom_row
    layout = row.layout()
    assert isinstance(layout, QHBoxLayout)
    assert layout.indexOf(panel.traded_section) >= 0
    assert layout.indexOf(panel.ideas_section) >= 0
    assert layout.indexOf(panel.traded_section) < layout.indexOf(panel.ideas_section)
    assert _is_descendant(panel.trades_table, panel.traded_section)
    assert _is_descendant(panel.ideas_note, panel.ideas_section)


# ==========================================================================
# 5. every table fills its cell
# ==========================================================================
def test_every_table_on_the_page_stretches_its_last_column(panel):
    from PySide6.QtWidgets import QHeaderView

    tables = panel.findChildren(QTableWidget)
    # ONE miss table (Day Recap step A), market calls, and entries and exits.
    assert len(tables) == 3, [table.objectName() for table in tables]
    for table in tables:
        header = table.horizontalHeader()
        assert header.stretchLastSection() is True, table
        for column in range(table.columnCount() - 1):
            assert header.sectionResizeMode(column) == QHeaderView.ResizeMode.ResizeToContents, (table, column)


def test_a_filled_table_still_stretches_its_last_column(panel):
    panel.render(
        _payload(
            rejected_that_worked=[_Row("NVDA"), _Row("AMD")],
            trades=[
                {
                    "trade_id": "t-1",
                    "symbol": "NVDA",
                    "direction": "LONG",
                    "quantity": 100,
                    "net_pnl": 240.0,
                    "status": "closed",
                    "opened_at": f"{SESSION}T07:05:00-07:00",
                }
            ],
        )
    )
    for table in panel.findChildren(QTableWidget):
        assert table.horizontalHeader().stretchLastSection() is True


@pytest.mark.parametrize("rows", ([], [_Row("NVDA"), _Row("AMD")]))
def test_the_long_walkaway_headers_are_not_clipped(panel, qapp, rows):
    """The TJ-2B long headers must fill their own table cells.

    Measured on 73d308a0 with the desk's own theme applied: "Against me first %"
    hints 273 px and the shared width rule's `MAX_COLUMN_WIDTH` clamped it to
    260, so the header was cut at both ends (it is centred, so a clip shows on
    each side and there is no ellipsis to warn anyone). The stylesheet is not
    applied inside the suite - it is global to the `QApplication` and would
    follow every later test - so the header font is enlarged here instead,
    which is the same fact: a title wider than the ceiling.

    A section that measures its CONTENTS is never narrower than its own hint.
    """
    from PySide6.QtGui import QFont

    table = panel.miss_table
    # Both long headers sit behind "More columns" since Day Recap step A.
    panel.more_columns_toggle.setChecked(True)
    header = table.horizontalHeader()
    font = QFont(header.font())
    font.setPointSize(16)
    font.setBold(True)
    header.setFont(font)

    panel.render(_payload(walkaway=_walkaway_day(rows)))
    panel.resize(1900, 1000)
    panel.show()
    qapp.processEvents()
    try:
        for column in range(table.columnCount()):
            text = table.horizontalHeaderItem(column).text()
            if text not in {"Held at close %", "Left on the table %"}:
                continue
            assert header.sectionSize(column) >= header.sectionSizeHint(column), (
                text,
                header.sectionSize(column),
                header.sectionSizeHint(column),
            )
    finally:
        panel.hide()


# ==========================================================================
# 6. the ratio is remembered
# ==========================================================================
def test_the_column_split_round_trips_through_the_saved_setting(qapp, clean_split_settings):
    from project_paths import get_local_setting, invalidate_local_settings_cache
    from ui.panels.day_review_panel import COLUMN_SPLIT_KEY

    first = _panel(qapp)
    try:
        # Wide enough that the two columns HAVE freedom: at their combined
        # minimum width a splitter ignores `setSizes`, which is Qt doing the
        # right thing and not a drag anyone could make.
        first.resize(1900, 1000)
        first.show()
        qapp.processEvents()
        first.columns.setSizes([1330, 570])
        first.columns.splitterMoved.emit(1330, 1)
        timer = first._split_save_timers[COLUMN_SPLIT_KEY]
        timer.stop()
        timer.timeout.emit()  # the debounce, fired without waiting 400 ms
        invalidate_local_settings_cache()
        saved = get_local_setting(COLUMN_SPLIT_KEY, None)
        assert isinstance(saved, list) and len(saved) == 2, saved
        assert saved[0] / max(1, sum(saved)) > 0.6, saved
    finally:
        first.hide()
        first.shutdown()
        first.deleteLater()
        qapp.processEvents()

    second = _panel(qapp)
    try:
        second.resize(1900, 1000)
        second.show()
        qapp.processEvents()
        sizes = second.columns.sizes()
        share = sizes[0] / max(1, sum(sizes))
        assert 0.66 <= share <= 0.74, sizes
    finally:
        second.hide()
        second.shutdown()
        second.deleteLater()
        qapp.processEvents()


# ==========================================================================
# 7. what TJ-1 already promised, re-asserted
# ==========================================================================
def test_the_page_is_still_one_scroll_area(panel):
    assert len(panel.findChildren(QScrollArea)) == 1


def test_the_one_spy_chart_is_still_built_on_first_need_and_reused(panel):
    from ui.widgets.candle_chart import CandleChart

    assert panel.findChildren(CandleChart) == [], "nothing expensive at construction"
    panel.render(_payload(spy_m5_bars=SPY_BARS))
    charts = panel.findChildren(CandleChart)
    assert len(charts) == 1
    first = charts[0]
    panel.render(_payload(spy_m5_bars=SPY_BARS[:6]))
    again = panel.findChildren(CandleChart)
    assert len(again) == 1 and again[0] is first


def test_the_forecast_collapses_to_three_lines(panel):
    from ui.panels.day_review_panel import FORECAST_COLLAPSED_LINES

    assert FORECAST_COLLAPSED_LINES == 3
    text = "\n".join(f"line {index}" for index in range(10))
    panel.render(_payload(forecast={"text": text, "source_model": "chatgpt"}))
    shown = panel.forecast_box.toPlainText()
    assert shown.count("\n") + 1 <= FORECAST_COLLAPSED_LINES, shown
    assert panel.forecast_toggle.isVisible() or panel.forecast_toggle.isVisibleTo(panel)
    panel.forecast_toggle.click()
    assert panel.forecast_box.toPlainText().strip().endswith("line 9")
