"""G-P2.1: AWAY Recap as a return surface (§8.3, decision 9).

The trader worked a Wave P1 build on 2026-08-26 and said of this page: "its just
hard to work with... i also cant even check charts from here. kinda useless."

Charting was wired the whole time - `symbolActivated` reaches the Alert Center's
snapshot popup - but the day's only two alerts were scanner status messages
(`Scanning ...`, `Learning ...`) with a blank symbol and side `WATCH`, so nothing
on the page was chartable, and nothing on the page said a row could be opened at
all. A blank symbol looked exactly like a symbol whose chart was broken.

Three presentation changes, and nothing else on the page moves. In particular
the Alert Center's backing list is NOT touched: `set_alerts` stays the one
reader here and the day's record keeps every row.

1. scanner status rows are hidden from the alerts table and COUNTED in one line;
   one click reveals them for the session;
2. a per-row `Chart` affordance, `Enter` on the selected row, and a header line
   that says so;
3. a symbol-less row renders distinctly and offers no chart action.
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

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _panel():
    from ui.panels.away_recap_panel import AwayRecapPanel

    return AwayRecapPanel()


#: The exact shape `BounceService._emit_assist_note` and the scanner-status
#: rows put on the Alert Center's backing list.
STATUS_ROW = {"symbol": "", "side": "WATCH", "trigger": "Scanning 1097 symbols", "time_text": "09:05:00"}
SECOND_STATUS_ROW = {"symbol": "", "side": "WATCH", "trigger": "Learning refresh", "time_text": "13:35:00"}
SYMBOL_ROW = {"symbol": "OKTA", "side": "LONG", "tier": "A", "trigger": "VWAP reclaim", "time_text": "09:31:00"}


def _day(*rows):
    return {"summary": "one day", "classified_alerts": list(rows)}


# -- 1. hide and count, never delete -----------------------------------------


def test_scanner_status_rows_are_hidden_and_counted(app):
    panel = _panel()
    panel._render(_day(STATUS_ROW, SYMBOL_ROW, SECOND_STATUS_ROW))

    assert panel.alerts.rowCount() == 1
    assert panel.alerts.item(0, 1).text() == "OKTA"
    # isVisibleTo, not isVisible: the panel itself is never shown in a headless
    # test, so isVisible() is False for everything and would assert nothing.
    assert panel.status_rows_toggle.isVisibleTo(panel) is True
    assert panel.status_rows_toggle.text() == "2 scanner status messages - show"


def test_one_status_row_reads_as_one_message(app):
    panel = _panel()
    panel._render(_day(STATUS_ROW, SYMBOL_ROW))

    assert panel.status_rows_toggle.text() == "1 scanner status message - show"


def test_the_filter_never_deletes_the_row_from_the_recap(app):
    """Hide and count. The day's record keeps every row it produced."""
    panel = _panel()
    day = _day(STATUS_ROW, SYMBOL_ROW)
    panel._render(day)

    assert len(panel._recap["classified_alerts"]) == 2
    assert panel._status_rows == [STATUS_ROW]


def test_one_click_reveals_them_for_the_session(app):
    panel = _panel()
    panel._render(_day(STATUS_ROW, SYMBOL_ROW, SECOND_STATUS_ROW))
    panel.status_rows_toggle.click()

    assert panel.alerts.rowCount() == 3
    symbols = [panel.alerts.item(row, 1).text() for row in range(3)]
    assert symbols == ["OKTA", "", ""], "the day's order is preserved within each group"
    assert "shown" in panel.status_rows_toggle.text()


def test_a_day_with_no_status_rows_shows_no_count_line(app):
    panel = _panel()
    panel._render(_day(SYMBOL_ROW))

    assert panel.status_rows_toggle.isVisibleTo(panel) is False
    assert panel.status_rows_toggle.text() == ""


def test_the_reveal_survives_a_later_refresh(app):
    """"For the session" means the next refresh does not re-hide them."""
    panel = _panel()
    panel._render(_day(STATUS_ROW, SYMBOL_ROW))
    panel.status_rows_toggle.click()
    panel._render(_day(STATUS_ROW, SYMBOL_ROW, SECOND_STATUS_ROW))

    assert panel.alerts.rowCount() == 3


def test_the_status_row_test_is_the_blank_symbol(app):
    from ui.panels.away_recap_panel import is_scanner_status_row

    assert is_scanner_status_row({"symbol": "", "side": "WATCH"}) is True
    assert is_scanner_status_row({"symbol": "   ", "side": "LONG"}) is True
    assert is_scanner_status_row({"symbol": "OKTA", "side": "WATCH"}) is False
    assert is_scanner_status_row({}) is True


# -- 2. the invitation -------------------------------------------------------


def test_the_page_says_how_to_open_a_row(app):
    panel = _panel()

    text = panel.chart_hint.text().lower()
    assert "enter" in text and "chart" in text


def test_every_chartable_row_carries_a_visible_chart_affordance(app):
    from ui.panels.away_recap_panel import CHART_CELL

    panel = _panel()
    panel._render(
        {
            "classified_alerts": [SYMBOL_ROW],
            "best_swings": [{"rank": 1, "symbol": "FROG", "side": "LONG", "text": "1. FROG"}],
        }
    )

    assert panel.alerts.item(0, panel.alerts.columnCount() - 1).text() == CHART_CELL
    assert panel.swings.item(0, panel.swings.columnCount() - 1).text() == CHART_CELL


def test_clicking_the_chart_cell_asks_the_host_to_chart_it(app):
    panel = _panel()
    panel._render(_day(SYMBOL_ROW))
    seen: list[str] = []
    panel.symbolActivated.connect(seen.append)

    panel._alert_cell_clicked(0, panel.alerts.columnCount() - 1)

    assert seen == ["OKTA"]


def test_clicking_another_cell_charts_nothing(app):
    """Single click selects. Only the chart cell is the button."""
    panel = _panel()
    panel._render(_day(SYMBOL_ROW))
    seen: list[str] = []
    panel.symbolActivated.connect(seen.append)

    panel._alert_cell_clicked(0, 1)

    assert seen == []


def test_enter_on_the_selected_row_charts_it(app):
    panel = _panel()
    panel._render(
        {
            "classified_alerts": [SYMBOL_ROW],
            "best_swings": [{"rank": 1, "symbol": "FROG", "side": "LONG", "text": "1. FROG"}],
        }
    )
    seen: list[str] = []
    panel.symbolActivated.connect(seen.append)

    for table in (panel.alerts, panel.swings):
        table.setCurrentCell(0, 0)
        press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        panel.eventFilter(table, press)

    assert seen == ["OKTA", "FROG"]


# -- 3. a symbol-less row is not a symbol row --------------------------------


def test_a_revealed_status_row_offers_no_chart_action(app):
    panel = _panel()
    panel._render(_day(STATUS_ROW))
    panel._reveal_status_rows()

    assert panel.alerts.item(0, panel.alerts.columnCount() - 1).text() == ""

    seen: list[str] = []
    panel.symbolActivated.connect(seen.append)
    panel._alert_cell_clicked(0, panel.alerts.columnCount() - 1)
    panel.alerts.setCurrentCell(0, 0)
    panel.eventFilter(
        panel.alerts,
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier),
    )

    assert seen == [], "a blank symbol must never open a chart for \"\""


def test_a_revealed_status_row_renders_distinctly(app):
    """It must not sit in the symbol table looking like a symbol row."""
    panel = _panel()
    panel._render(_day(STATUS_ROW, SYMBOL_ROW))
    panel._reveal_status_rows()

    symbol_item = panel.alerts.item(0, 5)
    status_item = panel.alerts.item(1, 5)

    assert status_item.font().italic() is True
    assert symbol_item.font().italic() is False
    assert status_item.foreground().color() != symbol_item.foreground().color()
    assert status_item.data(Qt.ItemDataRole.UserRole + 1) == "status"


def test_the_page_is_honest_when_there_is_no_recap(app):
    panel = _panel()
    panel._render({})

    assert panel.alerts.rowCount() == 0
    assert panel.swings.rowCount() == 0
    assert panel.status_rows_toggle.isVisibleTo(panel) is False
