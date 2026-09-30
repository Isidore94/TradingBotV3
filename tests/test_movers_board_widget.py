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
                "short": [_row("SINK", dip_score=-1.5, since_start_pct=-0.9)] if pullback else []},
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
    assert widget.mode == "pop"
    # Pop always lists longs and shorts together.
    assert _symbols(widget) == ["AAA", "BBB", "ZZZ"]
    rvol_col = [key for key, _h in widget.model._columns].index("rvol")
    assert widget.model.data(widget.model.index(1, rvol_col), Qt.ItemDataRole.DisplayRole) == "—"
    assert widget.model.data(widget.model.index(0, rvol_col), Qt.ItemDataRole.BackgroundRole) is not None
    assert widget.model.data(widget.model.index(1, rvol_col), Qt.ItemDataRole.BackgroundRole) is None
    assert "no pullback" in widget.banner.text()


def test_my_names_is_gone_and_a_saved_mine_mode_loads_as_pop(app, monkeypatch):
    # Trader 2026-09-29: "lets remove the 'my names' section of Movers".
    import project_paths
    from ui.widgets import movers_board
    from ui.widgets.movers_board import MoversBoard

    assert "mine" not in movers_board.MODES and "mine" not in movers_board.COLUMNS
    assert not hasattr(movers_board.movers_scan, "sort_mine")
    monkeypatch.setattr(project_paths, "get_local_setting",
                        lambda key, default=None: "mine" if key == "movers_board_mode" else default)
    widget = MoversBoard(persist=True)
    assert widget.mode == "pop"
    assert "mine" not in widget.mode_buttons
    assert not hasattr(widget, "side_button")
    widget.set_mode("mine")
    assert widget.mode == "pop"


def _section_symbols(section):
    return [row["symbol"] for row in section.visible_rows()]


def test_pullback_lights_both_dip_tables_under_pop_and_auto_switches_once(app):
    widget = _widget(app)
    widget.update_board(_board(pullback=False))
    widget.flush_pending_refresh()
    widget.update_board(_board(pullback=True))
    widget.flush_pending_refresh()
    # Pop, Dip-strong and Dip-weak all show at once.
    assert widget.mode == "pop"
    assert "●" in widget.mode_buttons["pop"].text()
    assert "PULLBACK" in widget.banner.text() and "-0.42%" in widget.banner.text()
    assert _symbols(widget) == ["AAA", "BBB", "ZZZ"]
    assert not widget.strong.isHidden() and not widget.weak.isHidden()
    assert not widget.strong.table.isHidden() and not widget.weak.table.isHidden()
    assert _section_symbols(widget.strong) == ["HOLD"]
    assert _section_symbols(widget.weak) == ["SINK"]
    assert widget.strong.title_label.text().startswith("Dip-strong")
    assert widget.weak.title_label.text().startswith("Dip-weak")


def test_dip_boxes_read_the_swing_lists_and_name_their_own_anchor(app):
    widget = _widget(app)
    board = _board(pullback=False)
    board["swing"] = {"long": [_row("LEAD", dip_score=1.1, since_start_pct=1.5)],
                      "short": [_row("LAG", dip_score=-0.9, since_start_pct=-1.2, held=True)]}
    board["swing_anchor"] = {
        "long": {"dt": "2026-09-22T12:25:00-04:00", "time": "12:25", "price": 398.0,
                 "kind": "swing"},
        "short": {"dt": "2026-09-22T10:55:00-04:00", "time": "10:55", "price": 404.5,
                  "kind": "swing"},
    }
    widget.update_board(board)
    widget.flush_pending_refresh()
    assert _section_symbols(widget.strong) == ["LEAD"]
    assert _section_symbols(widget.weak) == ["LAG"]
    strong, weak = widget.strong.title_label.text(), widget.weak.title_label.text()
    # Trader 2026-09-30: longs from SPY's high after the rip, shorts from its low.
    assert strong.startswith("Dip-strong") and "since SPY's high" in strong
    assert "not lit" not in strong and "low" not in strong
    assert weak.startswith("Dip-weak") and "since SPY's low" in weak and "high" not in weak
    assert strong != weak  # each box names its own swing
    # No major move yet: the low / high of day, said so.
    board["swing_anchor"]["long"]["kind"] = "hod"
    board["swing_anchor"]["short"]["kind"] = "lod"
    widget.update_board(board)
    widget.flush_pending_refresh()
    assert "high of day so far" in widget.strong.title_label.text()
    assert "low of day so far" in widget.weak.title_label.text()
    # Too early for either: the open.
    board["swing_anchor"]["long"]["kind"] = "open"
    board["swing_anchor"]["short"]["kind"] = "open"
    widget.update_board(board)
    widget.flush_pending_refresh()
    assert "since the open" in widget.strong.title_label.text()
    assert "since the open" in widget.weak.title_label.text()
    # A fresh anchor under 30 minutes old: last tick's is kept, said so.
    board["swing_anchor"]["long"].update(kind="swing", held_from_previous=True)
    widget.update_board(board)
    widget.flush_pending_refresh()
    assert "held" in widget.strong.title_label.text()
    assert "held" not in widget.weak.title_label.text()
    # No SPY bars at all: the boxes say so and stay empty.
    board["swing"] = {"long": [], "short": []}
    board["swing_anchor"] = {"long": None, "short": None}
    widget.update_board(board)
    widget.flush_pending_refresh()
    assert "no SPY bars" in widget.strong.title_label.text()


def test_dip_weak_rows_are_shorts_for_click_and_plus_focus(app):
    widget = _widget(app)
    widget.update_board(_board(pullback=True))
    widget.flush_pending_refresh()
    got, asked = [], []
    widget.symbolActivated.connect(lambda s, side: got.append((s, side)))
    widget.focusAddRequested.connect(lambda s, side: asked.append((s, side)))
    widget._on_clicked(widget.weak.proxy.index(0, 0))
    widget._on_clicked(widget.strong.proxy.index(0, 0))
    assert got == [("SINK", "SHORT"), ("HOLD", "LONG")]
    # One selection across the tables: picking a weak row clears the Pop selection.
    widget.table.selectRow(0)
    widget.weak.table.selectRow(0)
    assert not widget.table.selectionModel().hasSelection()
    widget.add_focus_button.click()
    assert asked == [("SINK", "short")]
    widget._hide_selected()
    assert _section_symbols(widget.weak) == []
    assert "lagging" in widget.weak.empty_label.text()


def test_bounce_titles_the_dip_tables_bounce(app):
    widget = _widget(app)
    board = _board(pullback=True)
    board["state"] = dict(board["state"], state="down_day", pullback=False, bounce=True)
    widget.update_board(board)
    widget.flush_pending_refresh()
    assert widget.strong.title_label.text().startswith("Bounce-strong")
    assert "low" in widget.weak.title_label.text()


def test_pop_symbol_cells_carry_the_side_colour_and_new_names_a_stronger_tint(app):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor

    from ui import theme

    widget = _widget(app)
    board = _tagged_board()
    board["pop"]["short"] = [_row("SINK", rank_change=1, streak=2, pop_score=-1.0)]
    widget.update_board(board)
    widget.flush_pending_refresh()
    col = [k for k, _h in widget.model._columns].index("symbol")
    background = [widget.model.data(widget.model.index(r, col), Qt.ItemDataRole.BackgroundRole)
                  for r in range(4)]
    assert all(color is not None for color in background)
    assert background[1].alphaF() > background[0].alphaF()  # AMD: first tick on the list
    assert background[0].rgb() == QColor(theme.color("long")).rgb()
    assert background[3].rgb() == QColor(theme.color("short")).rgb()  # SINK is a short
    # A single-side table tints only its new names.
    widget.model.set_rows(widget.model.rows(), "dip", "long")
    background = [widget.model.data(widget.model.index(r, col), Qt.ItemDataRole.BackgroundRole)
                  for r in range(3)]
    assert background[1] is not None
    assert background[0] is None and background[2] is None


def test_d1_trend_tags_unknown_and_failing_rows(app):
    from ui.widgets.movers_board import symbol_text

    widget = _widget(app)
    board = _tagged_board()
    for row, flag in zip(board["pop"]["long"], (True, None, True), strict=True):
        row.update(trend_long=flag, trend_short=False, last=100.0,
                   daily_bars=0 if flag is None else 200)
    widget.update_board(board)
    widget.flush_pending_refresh()
    assert _cell(widget, 0, "symbol") == "NVDA ER ▲2"
    assert _cell(widget, 1, "symbol") == "AMD new D1?"
    # A failing row says so; a short row reads its own side.
    assert symbol_text({"symbol": "XXX", "_side": "long", "trend_long": False,
                        "trend_short": True, "last": 100.0}) == "XXX D1✗"
    assert symbol_text({"symbol": "YYY", "_side": "short", "trend_long": False,
                        "trend_short": True, "last": 100.0}) == "YYY"
    assert symbol_text({"symbol": "ZZZ", "_side": "long", "trend_long": None,
                        "trend_short": None, "last": None}) == "ZZZ"


def test_unknown_state_banner_and_dip_hint(app):
    widget = _widget(app)
    widget.update_board({"state": {"state": "unknown"}, "pop": {}, "dip": {}})
    widget.flush_pending_refresh()
    assert "unknown" in widget.banner.text()
    # Pop mode always shows three boxes; unlit Dip tables sit empty under their titles.
    assert not widget.strong.isHidden() and not widget.weak.isHidden()
    assert not widget.strong.table.isHidden() and not widget.weak.table.isHidden()
    assert widget.strong.model.rowCount() == 0 and widget.weak.model.rowCount() == 0
    assert "not lit" in widget.strong.title_label.text()
    assert widget.strong.title_label.text().startswith("Dip-strong")
    assert widget.weak.title_label.text().startswith("Dip-weak")
    assert "No SPY pullback, bounce or rally" in widget.strong.title_label.toolTip()
    assert widget.strong.empty_label.isHidden()
    assert widget.main.title_label.text().startswith("Movers")
    layout = widget.layout()
    assert layout.stretch(layout.indexOf(widget.main)) == 2
    assert layout.stretch(layout.indexOf(widget.strong)) == 1
    assert layout.stretch(layout.indexOf(widget.weak)) == 1
    assert not widget.main.title_label.isHidden()


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

    for mode, main in (("pop", "move15_pct"), ("dip", "dip_score")):
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


def test_each_box_copy_button_puts_its_symbols_on_the_clipboard_in_view_order(app):
    """Trader 2026-09-29: one Copy per box, comma-joined for TC2000 / TradingView."""
    from PySide6.QtWidgets import QApplication

    widget = _widget(app)
    board = _board(pullback=True)
    board["pop"] = _sort_board()["pop"]
    board["pop"]["long"].append(_row(" aaa ", move15_pct=0.1))  # duplicate, lower case, padded
    board["pop"]["long"].append(_row("", move15_pct=0.05))  # blank symbol
    widget.update_board(board)
    widget.flush_pending_refresh()
    assert widget.mode == "pop"
    clipboard = QApplication.clipboard()
    for section in widget.sections:
        assert not section.copy_button.isHidden()
        assert section.copy_button.isEnabled()
    widget.main.copy_button.click()
    assert clipboard.text() == "AAA,ZZZ,BBB"
    assert widget.main.copy_button.text() == "Copied 3"
    widget.strong.copy_button.click()
    assert clipboard.text() == "HOLD"
    widget.weak.copy_button.click()
    assert clipboard.text() == "SINK"
    # A header sort changes the on-screen order, and the copy follows it.
    col = [k for k, _h in widget.model._columns].index("move15_pct")
    widget.table.horizontalHeader().sectionClicked.emit(col)
    widget.main.copy_button.click()
    assert clipboard.text() == "BBB,AAA,ZZZ"


def test_empty_box_copy_button_is_disabled_and_leaves_the_clipboard(app):
    from PySide6.QtWidgets import QApplication

    widget = _widget(app)
    widget.update_board(_board(pullback=False))
    widget.flush_pending_refresh()
    clipboard = QApplication.clipboard()
    clipboard.setText("KEEP")
    assert not widget.strong.copy_button.isEnabled()
    assert not widget.weak.copy_button.isEnabled()
    assert widget.strong.copy_symbols() == ""
    assert clipboard.text() == "KEEP"
    assert widget.main.copy_button.isEnabled()


# ------------------------------------------------------------------ M30 / Daily tabs
def _tf_board(tf="m30", session="2026-09-22"):
    as_of = f"{session}T11:30:00-04:00" if tf == "m30" else f"{session}T00:00:00-04:00"
    line = {"n": {"1": 4, "3": 10, "5": 0}, "mean_excess_pct": {"1": 0.2, "3": 0.8, "5": None},
            "beat_pct": {"1": 50.0, "3": 57.0, "5": None}, "lookback_sessions": 20}
    empty = {"n": {"1": 0, "3": 0, "5": 0}}
    return {
        "tf": tf, "session": session, "as_of": as_of,
        "scanned_at": f"{session}T12:00:05-04:00", "stale": False, "last_error": "",
        "state": {"state": "up_day", "spy_day_pct": 0.42},
        "pop": {"long": [_row("TFA", pop_score=2.0)], "short": [_row("TFZ", pop_score=-1.0)]},
        "swing": {"long": [_row("TFS", dip_score=1.1)], "short": [_row("TFW", dip_score=-0.7)]},
        "swing_anchor": {"long": {"dt": "2026-09-21T14:00:00-04:00", "date": "2026-09-21",
                                  "time": "14:00", "kind": "swing"},
                         "short": {"dt": f"{session}T00:00:00-04:00", "date": session,
                                   "time": "", "kind": "window"}},
        "summaries": {"pop": {"long": line, "short": empty},
                      "dip_strong": {"long": line}, "dip_weak": {"short": empty}},
        "offered": 900, "measured": 850,
    }


def test_m30_and_daily_tabs_show_the_three_boxes_from_their_boards(app):
    from PySide6.QtWidgets import QApplication

    from ui.widgets.movers_board import MODE_LABELS

    widget = _widget(app)
    assert [MODE_LABELS[m] for m in widget.mode_buttons] == ["Pop + Dip", "M30", "Daily"]
    widget.update_board(_board())
    widget.update_timeframe_board("m30", _tf_board("m30"))
    widget.update_timeframe_board("d1", _tf_board("d1"))
    widget.mode_buttons["m30"].click()
    widget.flush_pending_refresh()
    assert widget.mode == "m30"
    assert _symbols(widget) == ["TFA", "TFZ"]
    assert _section_symbols(widget.strong) == ["TFS"] and _section_symbols(widget.weak) == ["TFW"]
    assert not widget.strong.isHidden() and not widget.weak.isHidden()
    assert widget.main.title_label.text().startswith("M30 Movers · 3-bar moves at ")
    assert widget.strong.title_label.text().startswith("M30 Dip-strong · beating SPY since 9/21")
    assert [h for _k, h in widget.model._columns][:3] == ["Sym", "90m", "RVOL"]
    # The outcome line rides on each box title's hover.
    assert "Last 20 sessions: +0.8% vs SPY at 3d, 57% beat, n=10" in \
        widget.main.title_label.toolTip()
    assert widget.strong.title_label.toolTip() == \
        "Last 20 sessions: +0.8% vs SPY at 3d, 57% beat, n=10"
    assert widget.weak.title_label.toolTip() == "no results yet"
    assert "M30 scanned 9/22" in widget.banner.text()
    assert "850 of 900 measured" in widget.banner.text()
    widget.mode_buttons["d1"].click()
    widget.flush_pending_refresh()
    assert widget.main.title_label.text() == "Daily Movers · 3-day moves to the 9/22 close"
    assert widget.weak.title_label.text() == "Daily Dip-weak · lagging SPY since 9/22 high"
    assert [h for _k, h in widget.model._columns][:2] == ["Sym", "3d"]
    # Copy still copies the shown box.
    widget.main.copy_button.click()
    assert QApplication.clipboard().text() == "TFA,TFZ"
    widget.mode_buttons["pop"].click()
    assert _symbols(widget) == ["AAA", "BBB", "ZZZ"]


def test_timeframe_banner_says_not_scanned_yet_stale_and_failed(app):
    widget = _widget(app)
    widget.set_mode("m30")
    widget.flush_pending_refresh()
    assert "M30: not scanned yet" in widget.banner.text()
    assert widget.main.title_label.text() == "M30 Movers · not scanned yet"
    board = dict(_tf_board("d1"), stale=True, last_error="no SPY bars from Yahoo")
    widget.update_timeframe_board("d1", board)
    widget.set_mode("d1")
    widget.flush_pending_refresh()
    assert "stale" in widget.banner.text() and "last scan FAILED" in widget.banner.text()
    assert widget.meta_label.text() == "9/22 stale"


def test_pb_and_line_chips_emit_arm_requests_with_the_row_side_in_every_mode(app):
    widget = _widget(app)
    widget.resize(700, 400)
    widget.update_board(_board(pullback=True))
    widget.update_timeframe_board("d1", _tf_board("d1"))
    widget.flush_pending_refresh()
    asked = []
    widget.alertArmRequested.connect(lambda *args: asked.append(args))
    assert not widget.arm_buttons["pullback"].isEnabled()
    widget.weak.table.selectRow(0)  # SINK, a Dip-weak (short) row
    widget.arm_buttons["pullback"].click()
    widget.arm_buttons["d1_line_pullback"].click()
    assert asked == [("SINK", "short", "pullback"), ("SINK", "short", "d1_line_pullback")]
    widget.set_mode("d1")
    widget.flush_pending_refresh()
    widget.table.selectRow(0)
    widget.arm_buttons["pullback"].click()
    assert asked[-1] == ("TFA", "long", "pullback")


def test_row_menu_arms_and_shows_disarm_when_armed(app):
    widget = _widget(app)
    widget.update_timeframe_board("m30", _tf_board("m30"))
    widget.set_mode("m30")
    widget.flush_pending_refresh()
    armed = {"TFW": {"d1_line_pullback"}}
    widget.set_armed_kinds_provider(lambda symbol: armed.get(symbol, set()))
    index = widget.weak.proxy.index(0, 0)
    texts = [a.text() for a in widget.row_menu(index).actions()]
    assert "Arm D1 Pullback (fast) TFW (short)" in texts
    assert "✓ Disarm Pullback to D1 line TFW" in texts
    asked = []
    widget.alertArmRequested.connect(lambda *args: asked.append(args))
    action = next(a for a in widget.row_menu(index).actions() if a.text().startswith("✓"))
    action.trigger()
    assert asked == [("TFW", "short", "d1_line_pullback")]
    widget.weak.table.selectRow(0)
    assert widget.arm_buttons["d1_line_pullback"].text() == "Line ✓"
    assert widget.arm_buttons["pullback"].text() == "PB"


def test_a_pullback_pulls_to_pop_once_but_a_rally_never_leaves_m30(app):
    widget = _widget(app)
    widget.set_mode("m30")
    widget.update_board(_board(pullback=True))
    widget.flush_pending_refresh()
    assert widget.mode == "pop"
    # The trader goes back to M30; the same episode does not pull them away again.
    widget.mode_buttons["m30"].click()
    widget.update_board(_board(pullback=True))
    widget.flush_pending_refresh()
    assert widget.mode == "m30"
    rally = _board()
    rally["state"] = dict(rally["state"], state="up_day", rally=True,
                          start_dt="2026-09-22T12:30:00-04:00")
    widget.update_board(rally)
    widget.flush_pending_refresh()
    assert widget.mode == "m30"
