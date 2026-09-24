"""The Movers board in the Alert Center: placement, deep read, review menu, chart route."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def panel(tmp_path):
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui.panels.alert_center_panel import AlertCenterPanel

    return AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")


def test_movers_board_tops_the_column_and_deep_read_shows_the_strength_page(panel):
    column = panel.movers_column
    assert panel.tabs_row.widget(1) is column
    layout = column.layout()
    assert layout.itemAt(0).widget() is panel.movers_board
    assert layout.itemAt(1).widget() is panel.strength_page
    # Deep read is off by default: the page is hidden, never removed.
    assert panel.movers_board.deep_read_button.isChecked() is False
    assert panel.strength_page.isHidden()
    panel.movers_board.deep_read_button.click()
    assert not panel.strength_page.isHidden()
    panel.movers_board.deep_read_button.click()
    assert panel.strength_page.isHidden()


def test_review_doors_moved_into_the_movers_menu(panel, monkeypatch):
    assert panel.focus_strength.review_button.isHidden()
    assert panel.focus_strength.faded_button.isHidden()
    calls = []
    monkeypatch.setattr(panel, "review_focus_picks", lambda: calls.append("focus"))
    monkeypatch.setattr(panel, "review_faded_picks", lambda: calls.append("faded"))
    # Re-wire to the patched methods the way __init__ did.
    panel.movers_board.reviewAllRequested.disconnect()
    panel.movers_board.fadedReviewRequested.disconnect()
    panel.movers_board.reviewAllRequested.connect(panel.review_focus_picks)
    panel.movers_board.fadedReviewRequested.connect(panel.review_faded_picks)
    panel.movers_board.focus_review_action.setEnabled(True)
    panel.movers_board.faded_review_action.setEnabled(True)
    panel.movers_board.focus_review_action.trigger()
    panel.movers_board.faded_review_action.trigger()
    assert calls == ["focus", "faded"]


def test_review_menu_is_wired_to_the_panel_review_methods():
    source = (SCRIPTS_DIR / "ui" / "panels" / "alert_center_panel.py").read_text(encoding="utf-8")
    assert "self.movers_board.reviewAllRequested.connect(self.review_focus_picks)" in source
    assert "self.movers_board.fadedReviewRequested.connect(self.review_faded_picks)" in source


def test_a_movers_row_click_charts_in_the_review_pane(panel, monkeypatch):
    charted = []
    monkeypatch.setattr(
        panel, "chart_symbol",
        lambda symbol, side="", origin="": charted.append((symbol, side, origin)),
    )
    panel.movers_board.update_board({
        "state": {"state": "up_day"},
        "pop": {"long": [{"symbol": "NVDA", "move15_pct": 1.2}], "short": []},
    })
    panel.movers_board.flush_pending_refresh()
    panel.movers_board.table.clicked.emit(panel.movers_board.model.index(0, 0))
    assert charted == [("NVDA", "LONG", "the Movers board")]


def test_attach_movers_service_feeds_the_board(panel):
    from PySide6.QtCore import QObject, Signal

    class Service(QObject):
        moversChanged = Signal(dict)

        def board(self):
            return {"state": {"state": "unknown"}, "pop": {"long": [{"symbol": "AAA"}]}}

    service = Service()
    panel.attach_movers_service(service)
    panel.movers_board.flush_pending_refresh()
    assert [r["symbol"] for r in panel.movers_board.model.rows()] == ["AAA"]
    service.moversChanged.emit({"state": {}, "pop": {"long": [{"symbol": "BBB"}]}})
    panel.movers_board.flush_pending_refresh()
    assert [r["symbol"] for r in panel.movers_board.model.rows()] == ["BBB"]


def test_main_window_owns_one_movers_service_and_stops_it():
    source = (SCRIPTS_DIR / "ui" / "app.py").read_text(encoding="utf-8")
    assert source.count("MoversService(") == 1
    assert "self.trading_panel.alert_center.attach_movers_service(self.movers_service)" in source
    assert "self.movers_service.shutdown()" in source
