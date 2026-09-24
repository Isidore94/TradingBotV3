"""The Movers board widget: modes, side toggle, banner, auto-switch, review menu, clicks."""

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


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _row(symbol, **values):
    base = {"symbol": symbol, "move15_pct": 1.0, "move30_pct": 1.5, "day_pct": 2.0,
            "rvol": 2.0, "vs_spy15_pct": 0.8, "pop_score": 1.0, "dip_score": None,
            "since_start_pct": None, "note": "", "stale": False}
    base.update(values)
    return base


def _board(pullback=False, start="2026-09-22T10:15:00-04:00"):
    state = {"state": "up_day", "pullback": pullback, "bounce": False,
             "extreme_time": "10:15", "start_dt": start if pullback else "",
             "spy_from_extreme_pct": -0.42 if pullback else -0.05, "spy_day_pct": 0.6}
    return {
        "as_of": "2026-09-22T10:40:00-04:00",
        "state": state,
        "pop": {"long": [_row("AAA"), _row("BBB", rvol=None)], "short": [_row("ZZZ", move15_pct=-1.0)]},
        "dip": {"long": [_row("HOLD", dip_score=1.2, since_start_pct=0.3)] if pullback else [],
                "short": []},
        "mine": {"long": [_row("MYA", pop_score=0.2), _row("MYB", pop_score=2.0)], "short": []},
    }


def _widget(app):
    from ui.widgets.movers_board import MoversBoard

    board = MoversBoard(persist=False)
    board.resize(420, 400)
    return board


def _symbols(widget):
    return [row["symbol"] for row in widget.model.rows()]


def test_pop_mode_shows_long_rows_and_rvol_none_as_dash(app):
    from PySide6.QtCore import Qt

    widget = _widget(app)
    widget.update_board(_board())
    widget.flush_pending_refresh()
    assert widget.mode == "pop" and widget.side == "long"
    assert _symbols(widget) == ["AAA", "BBB"]
    rvol_col = [key for key, _h in widget.model._columns].index("rvol")
    assert widget.model.data(widget.model.index(1, rvol_col), Qt.ItemDataRole.DisplayRole) == "—"
    assert widget.model.data(widget.model.index(0, rvol_col), Qt.ItemDataRole.BackgroundRole) is not None
    assert widget.model.data(widget.model.index(1, rvol_col), Qt.ItemDataRole.BackgroundRole) is None
    assert "no pullback" in widget.banner.text()


def test_side_toggle_and_mine_sorted_by_score(app):
    widget = _widget(app)
    widget.update_board(_board())
    widget.flush_pending_refresh()
    widget.side_button.click()
    assert widget.side == "short" and _symbols(widget) == ["ZZZ"]
    widget.side_button.click()
    widget.mode_buttons["mine"].click()
    assert _symbols(widget) == ["MYB", "MYA"]


def test_pullback_lights_dip_and_auto_switches_once_per_episode(app):
    widget = _widget(app)
    widget.update_board(_board(pullback=False))
    widget.flush_pending_refresh()
    assert widget.mode == "pop"
    widget.update_board(_board(pullback=True))
    widget.flush_pending_refresh()
    assert widget.mode == "dip"
    assert "●" in widget.mode_buttons["dip"].text()
    assert "PULLBACK" in widget.banner.text() and "-0.42%" in widget.banner.text()
    assert _symbols(widget) == ["HOLD"]
    # The trader goes back to Pop; the same episode does not pull them away again.
    widget.mode_buttons["pop"].click()
    widget.update_board(_board(pullback=True))
    widget.flush_pending_refresh()
    assert widget.mode == "pop"
    # A new episode does.
    widget.update_board(_board(pullback=True, start="2026-09-22T12:00:00-04:00"))
    widget.flush_pending_refresh()
    assert widget.mode == "dip"


def test_unknown_state_banner_and_empty_dip_text(app):
    widget = _widget(app)
    widget.update_board({"state": {"state": "unknown"}, "pop": {}, "dip": {}, "mine": {}})
    widget.flush_pending_refresh()
    assert "unknown" in widget.banner.text()
    widget.mode_buttons["dip"].click()
    assert widget.model.rowCount() == 0
    assert "No SPY pullback" in widget.empty_label.text()


def test_row_click_emits_symbol_and_side(app):
    widget = _widget(app)
    widget.update_board(_board())
    widget.flush_pending_refresh()
    got = []
    widget.symbolActivated.connect(lambda s, side: got.append((s, side)))
    widget._on_clicked(widget.model.index(1, 0))
    assert got == [("BBB", "LONG")]


def test_review_menu_carries_counts_and_emits(app):
    from PySide6.QtCore import QObject, Signal

    class Store(QObject):
        focusChanged = Signal()

        def all_focus_by_category(self):
            return {"swing": {"long": ["A"], "short": []}, "m5": {"long": ["A", "B"], "short": []}}

        def faded_picks(self):
            return []

    widget = _widget(app)
    widget.set_focus_service(Store())
    assert widget.focus_review_action.text() == "Focus pick review (2)"
    assert widget.faded_review_action.text() == "Faded review (0)"
    assert widget.faded_review_action.isEnabled() is False
    fired = []
    widget.reviewAllRequested.connect(lambda: fired.append("focus"))
    widget.focus_review_action.trigger()
    assert fired == ["focus"]


def test_narrow_board_shows_fewer_columns_and_short_labels(app):
    from ui import theme

    widget = _widget(app)
    widget.update_board(_board())
    widget.flush_pending_refresh()
    widget.show()
    widget.resize(theme.px(170), 400)
    app.processEvents()
    narrow = widget.visible_column_count()
    widget.resize(600, 400)
    app.processEvents()
    wide = widget.visible_column_count()
    widget.hide()
    assert narrow < wide
    assert widget.minimumWidth() == theme.px(170)


def test_model_updates_in_place_without_reset(app):
    widget = _widget(app)
    resets = []
    widget.model.modelReset.connect(lambda: resets.append(1))
    widget.update_board(_board())
    widget.flush_pending_refresh()
    widget.update_board(_board())
    widget.flush_pending_refresh()
    assert resets == []
