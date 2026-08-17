"""R4 section 5: the already-checked-today marker on chart-opening tables.

Trader decision 2026-08-15: checked means RECORDED DECISIONS only (dislike,
favorite, veto, like, note) -- no view tracking, zero new capture. The marker is
presentation: it never filters, never reorders, and never feeds scoring.

The reordering half is the one worth guarding. A badge that quietly became a
ranking would be a scoring change wearing a display change's clothes, so the
sort role is asserted unchanged in both directions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

_QT = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")

from PySide6.QtCore import Qt  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = _QT.QApplication.instance() or _QT.QApplication([])
    yield app


COLUMNS = (("symbol", "Symbol"), ("excess", "Excess"))


def _model(rows=None):
    from ui.models.tracker_table_model import TrackerTableModel

    return TrackerTableModel(
        COLUMNS,
        rows or [{"symbol": "AAPL", "excess": 1.5}, {"symbol": "NVDA", "excess": -0.5}],
        numeric_keys={"excess"},
    )


def _display(model, row, col=0):
    return model.data(model.index(row, col), Qt.ItemDataRole.DisplayRole)


# --------------------------------------------------------------------------
# the marker
# --------------------------------------------------------------------------
def test_no_marker_before_anything_is_decided():
    model = _model()
    assert _display(model, 0) == "AAPL"


def test_a_decided_symbol_is_marked():
    model = _model()
    model.set_reviewed_symbols({"AAPL"})
    assert _display(model, 0) == "● AAPL"


def test_an_undecided_symbol_stays_bare():
    model = _model()
    model.set_reviewed_symbols({"AAPL"})
    assert _display(model, 1) == "NVDA"


def test_the_marker_is_case_insensitive():
    model = _model()
    model.set_reviewed_symbols({"aapl"})
    assert _display(model, 0) == "● AAPL"


def test_only_the_symbol_column_is_marked():
    """A '●' on a numeric cell would be a wrong number, not a badge."""
    model = _model()
    model.set_reviewed_symbols({"AAPL"})
    assert "●" not in str(_display(model, 0, col=1))


def test_a_decided_symbol_explains_itself():
    model = _model()
    model.set_reviewed_symbols({"AAPL"})
    tip = model.data(model.index(0, 0), Qt.ItemDataRole.ToolTipRole)
    assert "already recorded a decision" in tip


def test_clearing_the_set_removes_the_marker():
    model = _model()
    model.set_reviewed_symbols({"AAPL"})
    model.set_reviewed_symbols(set())
    assert _display(model, 0) == "AAPL"


# --------------------------------------------------------------------------
# it must not become a ranking
# --------------------------------------------------------------------------
def test_the_sort_value_is_unchanged_by_the_marker():
    from ui.models.tracker_table_model import SORT_ROLE

    model = _model()
    before = model.data(model.index(0, 0), SORT_ROLE)
    model.set_reviewed_symbols({"AAPL"})
    assert model.data(model.index(0, 0), SORT_ROLE) == before


def test_the_numeric_sort_value_is_unchanged_too():
    from ui.models.tracker_table_model import SORT_ROLE

    model = _model()
    before = model.data(model.index(0, 1), SORT_ROLE)
    model.set_reviewed_symbols({"AAPL"})
    assert model.data(model.index(0, 1), SORT_ROLE) == before


def test_the_row_payload_is_unchanged():
    """ROW_ROLE is what the chart-open handler reads. A '●' leaking into it
    would send a malformed symbol to the snapshot."""
    from ui.models.tracker_table_model import ROW_ROLE

    model = _model()
    model.set_reviewed_symbols({"AAPL"})
    assert model.data(model.index(0, 0), ROW_ROLE)["symbol"] == "AAPL"


def test_marking_the_same_set_twice_does_not_repaint():
    model = _model()
    model.set_reviewed_symbols({"AAPL"})
    emitted: list = []
    model.dataChanged.connect(lambda *a: emitted.append(1))
    model.set_reviewed_symbols({"AAPL"})
    assert emitted == []


def test_marking_a_new_set_repaints():
    model = _model()
    emitted: list = []
    model.dataChanged.connect(lambda *a: emitted.append(1))
    model.set_reviewed_symbols({"AAPL"})
    assert emitted


def test_an_empty_model_survives_being_marked():
    model = _model(rows=[])
    model.set_reviewed_symbols({"AAPL"})  # must not raise on an empty range


# --------------------------------------------------------------------------
# the Industry board's ETF cell
# --------------------------------------------------------------------------
def _industry_table(rows, reviewed):
    from PySide6.QtWidgets import QTableWidget

    from ui.panels.industry_panel import _fill_table

    columns = (("sector", "Sector"), ("etf", "ETF"), ("rs_score", "RS"))
    table = QTableWidget()
    table.setColumnCount(len(columns))
    _fill_table(table, columns, rows, reviewed_symbols=reviewed)
    return table


def test_the_industry_etf_cell_is_marked():
    table = _industry_table([{"sector": "Tech", "etf": "XLK", "rs_score": "1.2"}], {"XLK"})
    assert table.item(0, 1).text() == "● XLK"


def test_an_undecided_industry_etf_stays_bare():
    table = _industry_table([{"sector": "Tech", "etf": "XLK", "rs_score": "1.2"}], {"XLE"})
    assert table.item(0, 1).text() == "XLK"


def test_the_industry_sector_name_is_never_marked():
    """Only the cell that opens a chart carries the marker."""
    table = _industry_table([{"sector": "XLK", "etf": "XLK", "rs_score": "1.2"}], {"XLK"})
    assert table.item(0, 0).text() == "XLK"


def test_the_industry_marker_does_not_change_the_sort_value():
    table = _industry_table([{"sector": "Tech", "etf": "XLK", "rs_score": "1.2"}], {"XLK"})
    marked = table.item(0, 1)
    bare = _industry_table([{"sector": "Tech", "etf": "XLK", "rs_score": "1.2"}], set()).item(0, 1)
    assert marked._sort_value == bare._sort_value
