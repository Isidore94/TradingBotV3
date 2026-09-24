"""The Movers board widget: modes, side toggle, banner, auto-switch, review menu, clicks,
column sort and hide-for-today."""

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


def test_pop_mode_shows_both_sides_and_rvol_none_as_dash(app):
    from PySide6.QtCore import Qt

    widget = _widget(app)
    widget.update_board(_board())
    widget.flush_pending_refresh()
    assert widget.mode == "pop" and widget.side == "long"
    # Pop always lists longs and shorts together; its side toggle is hidden.
    assert _symbols(widget) == ["AAA", "BBB", "ZZZ"]
    assert widget.side_button.isHidden()
    rvol_col = [key for key, _h in widget.model._columns].index("rvol")
    assert widget.model.data(widget.model.index(1, rvol_col), Qt.ItemDataRole.DisplayRole) == "—"
    assert widget.model.data(widget.model.index(0, rvol_col), Qt.ItemDataRole.BackgroundRole) is not None
    assert widget.model.data(widget.model.index(1, rvol_col), Qt.ItemDataRole.BackgroundRole) is None
    assert "no pullback" in widget.banner.text()
    widget.mode_buttons["dip"].click()
    assert not widget.side_button.isHidden()


def test_side_toggle_and_mine_sorted_by_score(app):
    widget = _widget(app)
    widget.update_board(_board())
    widget.flush_pending_refresh()
    widget.mode_buttons["mine"].click()
    widget.side_button.click()
    assert widget.side == "short" and _symbols(widget) == []
    widget.side_button.click()
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
    widget._on_clicked(widget.model.index(2, 0))
    assert got == [("BBB", "LONG"), ("ZZZ", "SHORT")]


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


def test_header_shows_spy_bar_time_and_stale_flag(app):
    from ui.widgets import movers_board as mb

    widget = _widget(app)
    board = _board()
    board["as_of_stale"] = False
    widget.update_board(board)
    widget.flush_pending_refresh()
    clock = mb._local_clock(board["as_of"])
    assert widget.meta_label.text() == clock
    board = dict(board, as_of_stale=True)
    widget.update_board(board)
    widget.flush_pending_refresh()
    assert widget.meta_label.text() == f"{clock} stale"


def test_banner_counts_fresh_names(app):
    widget = _widget(app)
    board = dict(_board(), fresh=412, offered=504)
    widget.update_board(board)
    widget.flush_pending_refresh()
    assert "412 of 504 fresh" in widget.banner.text()


def _tagged_board():
    board = _board()
    board["pop"]["long"] = [
        _row("NVDA", er=True, rank_change=2, streak=3, group="Semis", hod_break=True,
             from_hod_atr=0.0, from_vwap_atr=1.1, ext_up=False),
        _row("AMD", rank_change=None, streak=1, group="Semis", hod_break=False,
             from_hod_atr=-0.4, from_vwap_atr=2.5, ext_up=True),
        _row("MU", rank_change=-1, streak=2, group="Semis", from_hod_atr=-0.8),
    ]
    board["groups"] = {"pop": {"long": [["Semis", 3]], "short": []}, "dip": {}}
    return board


def _cell(widget, row, key):
    from PySide6.QtCore import Qt

    col = [k for k, _h in widget.model._columns].index(key)
    return widget.model.data(widget.model.index(row, col), Qt.ItemDataRole.DisplayRole)


def test_sym_cell_carries_er_and_rank_change_and_lvl_tags(app):
    widget = _widget(app)
    widget.update_board(_tagged_board())
    widget.flush_pending_refresh()
    assert _cell(widget, 0, "symbol") == "NVDA ER ▲2"
    assert _cell(widget, 1, "symbol") == "AMD new"
    assert _cell(widget, 2, "symbol") == "MU ▼1"
    assert _cell(widget, 0, "lvl") == "HOD brk"
    assert _cell(widget, 1, "lvl") == "ext"
    assert _cell(widget, 2, "lvl") == "-0.8H"
    assert _cell(widget, 0, "group") == "Semis"
    assert "Groups: Semis ×3" in widget.groups_label.text()
    assert not widget.groups_label.isHidden()


def test_column_priority_sym_score_rvol_lvl_first(app):
    from ui.widgets.movers_board import COLUMNS

    for mode, main in (("pop", "move15_pct"), ("dip", "dip_score"), ("mine", "move15_pct")):
        keys = [k for k, _h in COLUMNS[mode]]
        assert keys[:4] == ["symbol", main, "rvol", "lvl"]


def test_plus_focus_emits_for_the_selected_row_only_on_click(app):
    widget = _widget(app)
    widget.update_board(_tagged_board())
    widget.flush_pending_refresh()
    asked = []
    widget.focusAddRequested.connect(lambda s, side: asked.append((s, side)))
    assert widget.add_focus_button.isEnabled() is False  # nothing selected
    widget.table.selectRow(1)
    assert widget.add_focus_button.isEnabled() is True
    assert asked == []  # selecting never adds
    widget.add_focus_button.click()
    assert asked == [("AMD", "long")]
    menu = widget.row_menu(widget.model.index(0, 0))
    actions = [a for a in menu.actions() if "Focus" in a.text()]
    actions[0].trigger()
    assert asked[-1] == ("NVDA", "long")


def test_model_updates_in_place_without_reset(app):
    widget = _widget(app)
    resets = []
    widget.model.modelReset.connect(lambda: resets.append(1))
    widget.update_board(_board())
    widget.flush_pending_refresh()
    widget.update_board(_board())
    widget.flush_pending_refresh()
    assert resets == []


def _day_board(state, as_of="2026-09-22T10:40:00-04:00"):
    board = _board()
    board["state"] = dict(board["state"], state=state)
    board["as_of"] = as_of
    return board


def test_side_follows_the_day_until_the_trader_taps_it(app):
    widget = _widget(app)
    widget.update_board(_day_board("down_day"))
    widget.flush_pending_refresh()
    assert widget.side == "short"
    # The trader taps Long; the same down day does not pull them back.
    widget.side_button.click()
    widget.update_board(_day_board("down_day", as_of="2026-09-22T10:45:00-04:00"))
    widget.flush_pending_refresh()
    assert widget.side == "long"
    # The day turns up: follow it again.
    widget.update_board(_day_board("up_day", as_of="2026-09-22T11:30:00-04:00"))
    widget.flush_pending_refresh()
    assert widget.side == "long"
    widget.update_board(_day_board("down_day", as_of="2026-09-22T13:00:00-04:00"))
    widget.flush_pending_refresh()
    assert widget.side == "short"


def test_side_does_not_follow_flat_or_unknown_days(app):
    widget = _widget(app)
    widget.set_side("short", user=True)
    for state in ("flat", "unknown"):
        widget.update_board(_day_board(state))
        widget.flush_pending_refresh()
        assert widget.side == "short"


def _view_symbols(widget):
    return [row["symbol"] for row in widget.visible_rows()]


def _sort_board():
    board = _board()
    board["pop"] = {
        "long": [_row("AAA", pop_score=3.0, move15_pct=0.9, rvol=1.2),
                 _row("BBB", pop_score=2.0, move15_pct=1.6, rvol=None)],
        "short": [_row("ZZZ", pop_score=-2.5, move15_pct=-1.4, rvol=4.0)],
    }
    return board


def test_pop_board_order_is_biggest_move_either_side(app):
    widget = _widget(app)
    widget.update_board(_sort_board())
    widget.flush_pending_refresh()
    assert _view_symbols(widget) == ["AAA", "ZZZ", "BBB"]
    # Each row keeps its own side for the Lvl cell, the chart click and +F.
    asked = []
    widget.focusAddRequested.connect(lambda s, side: asked.append((s, side)))
    menu = widget.row_menu(widget.model.index(1, 0))
    [a for a in menu.actions() if "Focus" in a.text()][0].trigger()
    assert asked == [("ZZZ", "short")]


def test_header_click_sorts_desc_then_asc_then_board_order(app):
    widget = _widget(app)
    widget.update_board(_sort_board())
    widget.flush_pending_refresh()
    col = [k for k, _h in widget.model._columns].index("move15_pct")
    widget.table.horizontalHeader().sectionClicked.emit(col)
    assert _view_symbols(widget) == ["BBB", "AAA", "ZZZ"]
    widget.table.horizontalHeader().sectionClicked.emit(col)
    assert _view_symbols(widget) == ["ZZZ", "AAA", "BBB"]
    widget.table.horizontalHeader().sectionClicked.emit(col)
    assert _view_symbols(widget) == ["AAA", "ZZZ", "BBB"]
    # The sort holds across a new board.
    widget.table.horizontalHeader().sectionClicked.emit(col)
    widget.update_board(_sort_board())
    widget.flush_pending_refresh()
    assert _view_symbols(widget) == ["BBB", "AAA", "ZZZ"]


def test_sort_keeps_unmeasured_last_both_ways_and_clicks_map_to_the_view(app):
    widget = _widget(app)
    widget.update_board(_sort_board())
    widget.flush_pending_refresh()
    col = [k for k, _h in widget.model._columns].index("rvol")
    widget.table.horizontalHeader().sectionClicked.emit(col)
    assert _view_symbols(widget) == ["ZZZ", "AAA", "BBB"]
    widget.table.horizontalHeader().sectionClicked.emit(col)
    assert _view_symbols(widget) == ["AAA", "ZZZ", "BBB"]
    got = []
    widget.symbolActivated.connect(lambda s, side: got.append((s, side)))
    widget._on_clicked(widget.proxy.index(1, 0))
    assert got == [("ZZZ", "SHORT")]


def test_hide_for_today_removes_the_row_and_unhide_brings_it_back(app):
    widget = _widget(app)
    widget.update_board(_sort_board())
    widget.flush_pending_refresh()
    assert widget.unhide_button.isHidden()
    menu = widget.row_menu(widget.model.index(1, 0))
    [a for a in menu.actions() if a.text().startswith("Hide ZZZ")][0].trigger()
    assert _view_symbols(widget) == ["AAA", "BBB"]
    assert not widget.unhide_button.isHidden() and "1" in widget.unhide_button.text()
    # A new tick the same day keeps it hidden.
    widget.update_board(_sort_board())
    widget.flush_pending_refresh()
    assert _view_symbols(widget) == ["AAA", "BBB"]
    # Delete on the selected row hides it too.
    widget.table.selectRow(0)
    widget._hide_selected()
    assert _view_symbols(widget) == ["BBB"]
    widget.unhide_button.click()
    assert _view_symbols(widget) == ["AAA", "ZZZ", "BBB"]
    assert widget.unhide_button.isHidden()


def test_hidden_rows_come_back_the_next_day(app):
    widget = _widget(app)
    widget.update_board(_sort_board())
    widget.flush_pending_refresh()
    widget.hide_row({"symbol": "AAA", "_side": "long"})
    assert "AAA" not in _view_symbols(widget)
    tomorrow = dict(_sort_board(), as_of="2026-09-23T09:40:00-04:00")
    widget.update_board(tomorrow)
    widget.flush_pending_refresh()
    assert "AAA" in _view_symbols(widget)


def test_hide_persists_through_the_local_setting(app, monkeypatch):
    import project_paths
    from ui.widgets import movers_board as mb

    saved = {}
    monkeypatch.setattr(project_paths, "save_local_setting", lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(project_paths, "get_local_setting", lambda k, d=None: saved.get(k, d))
    first = mb.MoversBoard(persist=True)
    first.update_board(_sort_board())
    first.flush_pending_refresh()
    first.hide_row({"symbol": "ZZZ", "_side": "short"})
    assert saved[mb.MOVERS_HIDDEN_SETTING] == {"day": "2026-09-22", "keys": ["ZZZ|short"]}
    second = mb.MoversBoard(persist=True)
    second.update_board(_sort_board())
    second.flush_pending_refresh()
    assert "ZZZ" not in _view_symbols(second)
