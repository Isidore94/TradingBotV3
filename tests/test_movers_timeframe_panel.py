"""Alert Center wiring for the Movers M30 / Daily tabs: the PB / Line chips call the
D1 menu's own toggles, the tf service feeds the tabs, +F on tf and Dip-box rows."""

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


class Focus:
    def __init__(self):
        self.added = []

    def add(self, symbol, side, category="m5", *, origin="", context=""):
        self.added.append((symbol, side, context))
        return True


def _panel(tmp_path):
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")
    panel.focus_service = Focus()
    return panel


def _row(symbol, **extra):
    base = {"symbol": symbol, "last": 105.0, "prev_high": 101.0, "prev_low": 97.0,
            "prev_session": "2026-09-21", "session_vwap": 103.0, "move15_pct": 1.0,
            "pop_score": 2.0, "dip_score": 1.0}
    base.update(extra)
    return base


def _tf_board(tf, rows_long=(), rows_short=(), session="2026-09-22"):
    return {"tf": tf, "session": session, "as_of": f"{session}T11:30:00-04:00",
            "state": {"state": "up_day"}, "pop": {"long": list(rows_long),
                                                  "short": list(rows_short)},
            "swing": {"long": [], "short": []}, "swing_anchor": {}}


def test_pb_and_line_clicks_call_the_d1_menu_toggles_with_the_row_side(tmp_path, app):
    panel = _panel(tmp_path)
    calls = []
    panel.toggle_chart_watch = lambda symbol, side, kind, *, source_text="": calls.append(
        ("watch", symbol, side, kind))
    panel.toggle_d1_event_watch = lambda symbol, kind, side="": calls.append(
        ("d1", symbol, side, kind))
    panel.movers_board.alertArmRequested.emit("TFW", "short", "pullback")
    panel.movers_board.alertArmRequested.emit("TFA", "long", "d1_line_pullback")
    assert calls == [("watch", "TFW", "SHORT", "pullback"),
                     ("d1", "TFA", "LONG", "d1_line_pullback")]
    # Nothing armed (stubbed): the status says so; an unknown kind does nothing.
    assert "not armed" in panel.movers_board.status_label.text()
    panel.movers_board.alertArmRequested.emit("TFA", "long", "sma_break")
    assert len(calls) == 2


def test_line_click_arms_then_disarms_and_the_menu_reads_it(tmp_path, app):
    panel = _panel(tmp_path)
    panel._arms_async = lambda: False  # arm on this thread
    panel.movers_board.alertArmRequested.emit("TFA", "long", "d1_line_pullback")
    assert "d1_line_pullback" in panel.armed_d1_event_kinds("TFA")
    assert panel.movers_board.status_label.text() == \
        "✓ TFA: Pullback to D1 line armed (long)."
    assert panel._movers_armed_kinds("TFA") == {"d1_line_pullback"}
    panel.movers_board.alertArmRequested.emit("TFA", "long", "d1_line_pullback")
    assert "d1_line_pullback" not in panel.armed_d1_event_kinds("TFA")
    assert panel.movers_board.status_label.text() == "TFA: Pullback to D1 line disarmed."


def test_attach_timeframe_service_feeds_the_tabs(tmp_path, app):
    from PySide6.QtCore import QObject, Signal

    class Service(QObject):
        timeframeBoardChanged = Signal(str, dict)

        def boards(self):
            return {"d1": _tf_board("d1", [_row("OLD")])}

    panel = _panel(tmp_path)
    service = Service()
    panel.attach_movers_timeframe_service(service)
    assert panel.movers_board.timeframe_board("d1")["pop"]["long"][0]["symbol"] == "OLD"
    service.timeframeBoardChanged.emit("m30", _tf_board("m30", [_row("NEW")]))
    assert panel.movers_board.timeframe_board("m30")["pop"]["long"][0]["symbol"] == "NEW"


def test_attach_passes_the_saved_daily_date_then_each_pick(tmp_path, app):
    from datetime import date

    from PySide6.QtCore import QObject, Signal

    class Service(QObject):
        timeframeBoardChanged = Signal(str, dict)

        def __init__(self):
            super().__init__()
            self.calls = []

        def boards(self):
            return {}

        def set_d1_since(self, day, *, rebuild=True):
            self.calls.append((day, rebuild))

    panel = _panel(tmp_path)
    panel.movers_board._d1_since = date(2026, 9, 2)  # as restored from the setting
    service = Service()
    panel.attach_movers_timeframe_service(service)
    assert service.calls == [(date(2026, 9, 2), False)]  # stored before the first scan
    panel.movers_board.d1SinceChanged.emit(date(2026, 9, 8))
    assert service.calls[-1] == (date(2026, 9, 8), True)


def test_plus_focus_on_an_m30_row_needs_todays_m5_row(tmp_path, app):
    # Reviewer 2026-09-29: never gate on the M30 board's own session, levels or 12:00 last.
    panel = _panel(tmp_path)
    panel.movers_board.update_timeframe_board("m30", _tf_board("m30", [_row("TFA")]))
    panel.movers_board.set_mode("m30")
    panel.movers_board.focusAddRequested.emit("TFA", "long")
    assert panel.focus_service.added == []
    assert "M30 row" in panel.movers_board.status_label.text()
    # On today's M5 board: the M5 row's live levels decide.
    panel.movers_board.update_board({"as_of": "2026-09-22T13:40:00-04:00", "state": {},
                                     "pop": {"long": [_row("TFA")], "short": []}})
    panel.movers_board.focusAddRequested.emit("TFA", "long")
    assert [a[:2] for a in panel.focus_service.added] == [("TFA", "long")]
    assert panel.focus_service.added[0][2].startswith("movers 15m")


def test_plus_focus_refuses_a_stale_m30_board_row(tmp_path, app):
    panel = _panel(tmp_path)
    stale = _tf_board("m30", [_row("OLD", prev_session="2026-09-21")], session="2026-09-22")
    panel.movers_board.update_timeframe_board("m30", dict(stale, stale=True))
    panel.movers_board.update_board({"as_of": "2026-09-24T10:40:00-04:00", "state": {},
                                     "pop": {"long": [], "short": []}})
    panel.movers_board.set_mode("m30")
    panel.movers_board.focusAddRequested.emit("OLD", "long")
    assert panel.focus_service.added == []
    assert "✕ OLD" in panel.movers_board.status_label.text()


def test_plus_focus_on_a_daily_row_refuses_without_m5_levels(tmp_path, app):
    panel = _panel(tmp_path)
    panel.movers_board.update_timeframe_board("d1", _tf_board("d1", [_row("DLY")]))
    panel.movers_board.set_mode("d1")
    panel.movers_board.focusAddRequested.emit("DLY", "long")
    assert panel.focus_service.added == []
    assert "Daily row" in panel.movers_board.status_label.text()
    # On today's M5 board too: the M5 row's live levels decide.
    panel.movers_board.update_board({"as_of": "2026-09-22T10:40:00-04:00", "state": {},
                                     "pop": {"long": [_row("DLY")], "short": []}})
    panel.movers_board.focusAddRequested.emit("DLY", "long")
    assert [a[:2] for a in panel.focus_service.added] == [("DLY", "long")]


def test_plus_focus_finds_a_dip_box_row_on_the_m5_board(tmp_path, app):
    panel = _panel(tmp_path)
    panel.movers_board.update_board({
        "as_of": "2026-09-22T10:40:00-04:00", "state": {}, "pop": {}, "dip": {}, "rip": {},
        "swing": {"long": [_row("BOX")], "short": []}})
    panel.movers_board.focusAddRequested.emit("BOX", "long")
    assert "no longer on the board" not in panel.movers_board.status_label.text()
    assert [a[:2] for a in panel.focus_service.added] == [("BOX", "long")]
