"""Strength board: click-to-sort columns and chart-on-selection (2026-08-19).

Both are trader requests against a board that was "just a lot of picks". What
these tests protect is the part that is easy to get wrong and expensive to
notice on a live morning:

- sorting is PRESENTATION. It must never call the service, so it can never
  cause a refetch — the board's whole data budget is a 15-minute batched
  yfinance pull and a header click has to stay free;
- an "Add to Focus" button must stay attached to its own symbol after a
  re-sort, and every add must still re-run the adoption gate at click time;
- a blank cell is an absence, not a small number: it sorts last whichever way
  the arrow points;
- selecting a row charts it exactly once, through the desk's existing popup,
  and a refresh that keeps the same row selected is not a new request.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Qt, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.panels.strength_board_panel import (  # noqa: E402
    StrengthBoardPanel,
    sort_rows,
)

LONGS = [
    {"symbol": "BBB", "strength": 3.0, "day_pct": 1.0, "vwap_distance_pct": 0.4, "last": 50.0,
     "prev_high": 10.0, "prev_low": 5.0, "session_vwap": 20.0},
    {"symbol": "AAA", "strength": 9.0, "day_pct": -2.0, "vwap_distance_pct": 1.5, "last": 10.0,
     "prev_high": 5.0, "prev_low": 1.0, "session_vwap": 6.0},
    {"symbol": "CCC", "strength": 5.0, "day_pct": None, "vwap_distance_pct": 0.1, "last": 30.0,
     "prev_high": 8.0, "prev_low": 2.0, "session_vwap": 9.0},
]


class _Service(QObject):
    """The board service, minus everything but the cached rows."""

    boardChanged = Signal(dict)
    statusChanged = Signal(str)

    def __init__(self):
        super().__init__()
        self.board_calls = 0
        self.refresh_calls = 0

    def board(self):
        self.board_calls += 1
        return {"long": LONGS, "short": []}

    def refresh_now(self):  # pragma: no cover - must never be reached by a sort
        self.refresh_calls += 1

    def status_text(self):
        return "test board"


def _panel(monkeypatch):
    QApplication.instance() or QApplication([])
    panel = StrengthBoardPanel()
    panel.set_board({"long": LONGS, "short": []})
    return panel


class TestTheSortIsPure:
    def test_a_column_sorts_on_its_number_not_its_text(self):
        ordered = sort_rows(LONGS, 1, descending=True)
        assert [row["symbol"] for row in ordered] == ["AAA", "CCC", "BBB"]
        ordered = sort_rows(LONGS, 1, descending=False)
        assert [row["symbol"] for row in ordered] == ["BBB", "CCC", "AAA"]

    def test_negative_percentages_sort_as_numbers(self):
        """"-2.00%" sorts above "+1.00%" as text and below it as a number."""
        ordered = sort_rows(LONGS, 2, descending=True)
        assert [row["symbol"] for row in ordered][0] == "BBB"

    def test_a_blank_cell_sorts_last_in_both_directions(self):
        """CCC has no day %. It is an absence, not the smallest number."""
        for descending in (True, False):
            ordered = sort_rows(LONGS, 2, descending=descending)
            assert ordered[-1]["symbol"] == "CCC", descending

    def test_the_symbol_column_sorts_alphabetically(self):
        assert [row["symbol"] for row in sort_rows(LONGS, 0, descending=False)] == [
            "AAA",
            "BBB",
            "CCC",
        ]

    def test_an_out_of_range_column_changes_nothing(self):
        assert sort_rows(LONGS, 99, descending=True) == LONGS


class TestTheHeaderClicks:
    def test_clicking_a_header_reorders_the_visible_rows(self, monkeypatch):
        panel = _panel(monkeypatch)
        table = panel.longs.table
        first_before = table.item(0, 0).text()

        panel.longs._on_header_clicked(0)  # Symbol, A-Z
        assert [table.item(row, 0).text() for row in range(table.rowCount())] == [
            "AAA",
            "BBB",
            "CCC",
        ]
        assert first_before == "AAA", "the default order is strongest-first"

    def test_clicking_the_same_header_twice_flips_it(self, monkeypatch):
        panel = _panel(monkeypatch)
        panel.longs._on_header_clicked(0)
        panel.longs._on_header_clicked(0)
        table = panel.longs.table
        assert [table.item(row, 0).text() for row in range(table.rowCount())] == [
            "CCC",
            "BBB",
            "AAA",
        ]

    def test_the_indicator_says_which_way(self, monkeypatch):
        panel = _panel(monkeypatch)
        header = panel.longs.table.horizontalHeader()
        assert header.isSortIndicatorShown()
        panel.longs._on_header_clicked(2)
        assert header.sortIndicatorSection() == 2
        assert header.sortIndicatorOrder() == Qt.DescendingOrder
        panel.longs._on_header_clicked(2)
        assert header.sortIndicatorOrder() == Qt.AscendingOrder

    def test_the_default_order_states_each_sides_own_ranking(self, monkeypatch):
        """Longs rank strength descending, shorts ascending - the board's own
        order, now shown rather than left for the trader to re-derive."""
        panel = _panel(monkeypatch)
        assert panel.longs.sort_state() == (1, True)
        assert panel.shorts.sort_state() == (1, False)

    def test_the_button_column_sorts_nothing(self, monkeypatch):
        panel = _panel(monkeypatch)
        before = panel.longs.sort_state()
        # The button sits AFTER the last data column, whatever that index is -
        # V1 added three columns, and pinning the number here would mean editing
        # this test every time the board gains one.
        from ui.panels.strength_board_panel import _COLUMNS

        panel.longs._on_header_clicked(len(_COLUMNS))
        assert panel.longs.sort_state() == before

    def test_sorting_never_touches_the_service(self, monkeypatch):
        """A header click must not cost a fetch."""
        QApplication.instance() or QApplication([])
        service = _Service()
        panel = StrengthBoardPanel(service=service)
        service.board_calls = 0
        panel.longs._on_header_clicked(2)
        panel.longs._on_header_clicked(3)
        assert service.refresh_calls == 0
        assert service.board_calls == 0


class TestTheAddButtonFollowsItsRow:
    def test_the_button_still_adds_its_own_symbol_after_a_resort(self, monkeypatch):
        panel = _panel(monkeypatch)
        requested: list[tuple[str, str]] = []
        panel.longs.addRequested.connect(lambda symbol, side: requested.append((symbol, side)))

        panel.longs._on_header_clicked(0)  # A-Z: AAA, BBB, CCC
        table = panel.longs.table
        row_of_ccc = [
            index for index in range(table.rowCount()) if table.item(index, 0).text() == "CCC"
        ][0]
        from ui.panels.strength_board_panel import _COLUMNS

        table.cellWidget(row_of_ccc, len(_COLUMNS)).click()

        assert requested == [("CCC", "long")]

    def test_an_add_still_runs_the_gate_at_click_time(self, monkeypatch):
        """Sorting must not slip a row past packet R2 Part A."""
        QApplication.instance() or QApplication([])
        service = _Service()

        class _Focus:
            def __init__(self):
                self.added = []

            def add(self, symbol, side, category, origin="", context=""):
                self.added.append(symbol)
                return True

        focus = _Focus()
        panel = StrengthBoardPanel(service=service, focus_service=focus)
        panel.longs._on_header_clicked(0)

        gated: list[str] = []
        real_gate = panel._gate_row

        def _spy(row, side):
            gated.append(str(row.get("symbol")))
            return real_gate(row, side)

        panel._gate_row = _spy  # type: ignore[method-assign]
        panel._add_one("AAA", "long")
        assert gated == ["AAA"], "the gate ran for the clicked symbol"


class TestChartOnSelection:
    def test_selecting_a_row_charts_it_once(self, monkeypatch):
        panel = _panel(monkeypatch)
        charted: list[str] = []
        panel.symbolActivated.connect(lambda symbol, _side: charted.append(symbol))

        panel.longs.table.selectRow(0)
        assert charted == [panel.longs.table.item(0, 0).text()]

        # Selecting the same row again is not a new request.
        panel.longs.table.selectRow(0)
        assert len(charted) == 1

    def test_selecting_on_one_side_releases_the_other(self, monkeypatch):
        panel = _panel(monkeypatch)
        panel.set_board({"long": LONGS, "short": [dict(LONGS[0], symbol="ZZZ")]})
        charted: list[str] = []
        panel.symbolActivated.connect(lambda symbol, _side: charted.append(symbol))

        panel.longs.table.selectRow(0)
        panel.shorts.table.selectRow(0)

        assert charted[-1] == "ZZZ"
        assert panel.longs.selected_symbol() == ""

    def test_a_refresh_keeps_the_charted_row_selected(self, monkeypatch):
        """The chart must not wander to whatever name lands on that row."""
        panel = _panel(monkeypatch)
        panel.longs.table.selectRow(0)
        charted_first = panel.longs.selected_symbol()

        charted: list[str] = []
        panel.symbolActivated.connect(lambda symbol, _side: charted.append(symbol))
        panel.set_board({"long": list(reversed(LONGS)), "short": []})

        assert panel.longs.selected_symbol() == charted_first
        assert charted == [], "a refresh is not a new chart request"

    def test_a_symbol_that_leaves_the_board_does_not_repoint_the_chart(self, monkeypatch):
        panel = _panel(monkeypatch)
        panel.longs.table.selectRow(0)
        charted: list[str] = []
        panel.symbolActivated.connect(lambda symbol, _side: charted.append(symbol))

        panel.set_board({"long": [LONGS[0]], "short": []})
        assert charted == []

    def test_double_click_still_charts(self, monkeypatch):
        panel = _panel(monkeypatch)
        charted: list[str] = []
        panel.symbolActivated.connect(lambda symbol, _side: charted.append(symbol))
        panel.longs._on_double_click(1, 0)
        assert charted == [panel.longs.table.item(1, 0).text()]


# --------------------------------------------------------------------------
# The board moved into the Desk's Strength window (trader, 2026-08-31), and the
# RS/RW half it used to carry went with the page.
#
# That half existed for one reason (trader, 2026-08-21): the board says who is
# strong on the day, the RS/RW read says who is strong RELATIVE to SPY, and
# flipping between PAGES to compare them was the friction. The board now lives
# in the alert column, where the Alert Center's own RS/RW Board tab is one
# tab-click away in that same column - so a second RrsSnapshotWidget here would
# have been a duplicate view of one payload, six inches from the original.
#
# What did NOT change: the tape, its owner, the rrsSnapshotChanged signal, and
# the Alert Center's RS/RW Board tab. Only this second VIEW retired.
# --------------------------------------------------------------------------
def test_the_board_no_longer_carries_its_own_rs_rw_view():
    from ui.panels.strength_board_panel import StrengthBoardPanel
    from ui.widgets.rrs_snapshot import RrsSnapshotWidget

    panel = StrengthBoardPanel()
    assert panel.findChild(RrsSnapshotWidget) is None
    assert not hasattr(panel, "rrs_snapshot")
    assert not hasattr(panel, "update_rrs_snapshot")


def test_the_alert_center_still_owns_the_rs_rw_board():
    """The surviving view, so the read did not leave the desk with the page."""
    source = (SCRIPTS_DIR / "ui" / "panels" / "alert_center_panel.py").read_text(
        encoding="utf-8"
    )
    assert "self.rrs_snapshot = RrsSnapshotWidget()" in source
    assert 'self.tabs.addTab(board_tab, "RS/RW Board")' in source
    assert "service.rrsSnapshotChanged.connect(self.rrs_snapshot.update_snapshot)" in source


def test_the_two_sides_stack_vertically_for_a_narrow_column():
    """Two five-column tables side by side were readable on a full-width page.
    In the alert column they are not, so the splitter runs top-to-bottom and
    either side can be given the whole section."""
    from PySide6.QtCore import Qt

    from ui.panels.strength_board_panel import StrengthBoardPanel

    panel = StrengthBoardPanel()
    assert panel.sides.orientation() == Qt.Orientation.Vertical
    assert panel.sides.count() == 2
    assert panel.sides.widget(0) is panel.longs
    assert panel.sides.widget(1) is panel.shorts
    assert panel.sides.childrenCollapsible() is True
