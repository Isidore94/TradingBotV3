"""V1 item 1 - the board shows what nearly qualified, and can hide it again.

Decision 0010: a display filter is not a suppression. The board keeps every row
in the top strength slice; the ones that miss one of the trader's filters are
GREYED and name what they missed, and the "TC2000 parity" toggle - default ON,
because the trader reads this beside their TC2000 screen - hides them for a
line-by-line comparison.

Nothing here mutes, withholds, parks or writes anything.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _rows():
    return [
        {"symbol": "PICK", "strength": 90.0, "rvol": 2.4, "last": 50.0,
         "sma200_d1": 40.0, "sma100_d1": 45.0, "failed_floors": []},
        {"symbol": "NEARLY", "strength": 88.0, "rvol": 0.4, "last": 50.0,
         "sma200_d1": 40.0, "sma100_d1": 45.0,
         "failed_floors": ["not in the busier half by RVOL"]},
        {"symbol": "CHEAP", "strength": 87.0, "rvol": 3.0, "last": 2.0,
         "sma200_d1": 1.0, "sma100_d1": 1.5,
         "failed_floors": ["price $2.00 is not over $5"]},
    ]


def test_parity_is_on_by_default_and_shows_only_the_picks(qapp):
    from ui.panels.strength_board_panel import _SideTable

    table = _SideTable("long")
    table.set_rows(_rows())

    shown = [table.table.item(index, 0).text() for index in range(table.table.rowCount())]
    assert shown == ["PICK"]
    assert "2 below the filters" in table._title.text()


def test_turning_it_off_shows_every_row_and_names_what_each_missed(qapp):
    from ui.panels.strength_board_panel import _SideTable

    table = _SideTable("long")
    table.set_rows(_rows())
    table.set_parity_only(False)

    shown = [table.table.item(index, 0).text() for index in range(table.table.rowCount())]
    assert sorted(shown) == ["CHEAP", "NEARLY", "PICK"]

    tips = {
        table.table.item(index, 0).text(): table.table.item(index, 0).toolTip()
        for index in range(table.table.rowCount())
    }
    assert "busier half by RVOL" in tips["NEARLY"]
    assert "not over $5" in tips["CHEAP"]
    assert tips["PICK"] == "", "a pick has nothing to explain"


def test_the_toggle_hides_and_never_deletes(qapp):
    """The rows come straight back, because they were never removed."""
    from ui.panels.strength_board_panel import _SideTable

    table = _SideTable("long")
    table.set_rows(_rows())
    assert table.table.rowCount() == 1

    table.set_parity_only(False)
    assert table.table.rowCount() == 3

    table.set_parity_only(True)
    assert table.table.rowCount() == 1
    # And the data behind them is untouched throughout.
    assert len(table._rows) == 3


def test_the_new_columns_show_the_traders_own_numbers(qapp):
    from ui.panels.strength_board_panel import _COLUMNS, _SideTable

    assert "RVOL" in _COLUMNS
    assert "200 SMA" in _COLUMNS and "100 SMA" in _COLUMNS

    table = _SideTable("long")
    table.set_rows(_rows())
    header = {name: index for index, name in enumerate(_COLUMNS)}
    assert table.table.item(0, header["RVOL"]).text() == "2.40"
    assert table.table.item(0, header["200 SMA"]).text() == "40.00"


def test_an_unmeasured_number_reads_blank_rather_than_zero(qapp):
    from ui.panels.strength_board_panel import _COLUMNS, _SideTable

    table = _SideTable("long")
    table.set_rows([{"symbol": "NEW", "strength": 5.0, "rvol": None,
                     "sma200_d1": None, "failed_floors": []}])
    header = {name: index for index, name in enumerate(_COLUMNS)}
    assert table.table.item(0, header["RVOL"]).text() == "—"
    assert table.table.item(0, header["200 SMA"]).text() == "—"


def test_the_panel_switch_drives_both_sides(qapp):
    from ui.panels.strength_board_panel import StrengthBoardPanel

    panel = StrengthBoardPanel()
    assert panel.parity_toggle.isChecked() is True

    panel.longs.set_rows(_rows())
    panel.shorts.set_rows(_rows())
    assert panel.longs.table.rowCount() == 1
    assert panel.shorts.table.rowCount() == 1

    panel.parity_toggle.setChecked(False)
    assert panel.longs.table.rowCount() == 3
    assert panel.shorts.table.rowCount() == 3
