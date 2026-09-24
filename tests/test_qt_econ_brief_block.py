"""The Mentor's "Today's news & econ" block: shown once a session, redrawn on paste."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

SESSION = "2026-09-25"


def _view(origin="night", origin_text="night AI summary of the last brief", lines=None, today=None):
    return {
        "session": SESSION,
        "origin": origin,
        "origin_text": origin_text,
        "brief_session": "2026-09-24",
        "summary_lines": lines if lines is not None else ["Durable goods at 8:30 a.m. can move rates."],
        "today": today
        if today is not None
        else [
            {"date": SESSION, "time_et": "08:30", "label": "August durable goods orders"},
            {"date": SESSION, "time_et": "10:00", "label": "Michigan sentiment"},
        ],
        "week": [
            {"date": "2026-09-29", "time_et": "10:00", "label": "JOLTS"},
            {"date": "2026-09-30", "time_et": "", "label": "PCE"},
        ],
        "unread_lines": 0,
        "note": "",
    }


# ---------------------------------------------------------------------------
# the text
# ---------------------------------------------------------------------------
def test_the_block_lists_the_summary_today_in_et_and_pt_and_the_week():
    from ui.widgets.econ_brief_block import format_view

    text = format_view(_view())
    assert "• Durable goods at 8:30 a.m. can move rates." in text
    assert "08:30 ET (5:30 PT) — August durable goods orders" in text
    assert "10:00 ET (7:00 PT) — Michigan sentiment" in text
    assert "Tue 09-29 10:00 ET — JOLTS" in text
    assert "Wed 09-30 (time not given) — PCE" in text


def test_no_brief_is_said_plainly():
    from ui.widgets.econ_brief_block import format_view

    text = format_view({"session": SESSION, "note": "No brief pasted.", "today": [], "week": []})
    assert text == "No brief pasted."


def test_unread_calendar_lines_are_counted():
    from ui.widgets.econ_brief_block import format_view

    view = _view()
    view["unread_lines"] = 2
    assert "2 calendar lines not read." in format_view(view)


# ---------------------------------------------------------------------------
# the popup
# ---------------------------------------------------------------------------
@pytest.fixture()
def pane():
    from ui.widgets.alert_chart_review import AlertChartReview

    widget = AlertChartReview()
    yield widget
    widget.mentor_popup.hide()
    widget.deleteLater()


def test_the_block_can_open_the_popup_with_no_prompt_due(pane):
    pane.show_econ_brief(_view())
    assert pane.econ_block.isVisibleTo(pane.mentor_popup)
    assert pane.mentor_popup.isVisible()
    assert not pane.mentor_card.isVisibleTo(pane.mentor_popup)
    assert "Michigan sentiment" in pane.econ_block.text()
    assert "night AI summary" in pane.econ_block.origin_label.text()


def test_a_paste_redraws_the_block_in_place_without_opening_the_popup(pane):
    pane.show_econ_brief(_view())
    pane.mentor_popup.hide()
    pane.update_econ_brief(
        _view(origin="today_brief", origin_text="from today's brief", lines=["Watch, in order: 10Y → Brent"])
    )
    assert not pane.mentor_popup.isVisible()
    assert "from today's brief" in pane.econ_block.origin_label.text()
    assert "Watch, in order" in pane.econ_block.text()


def test_hide_closes_the_popup_when_no_prompt_is_up(pane):
    pane.show_econ_brief(_view())
    pane.econ_block.hide_button.click()
    assert not pane.econ_block.isVisibleTo(pane.mentor_popup)
    assert not pane.mentor_popup.isVisible()


# ---------------------------------------------------------------------------
# once a session, across a restart
# ---------------------------------------------------------------------------
class _Review:
    def __init__(self):
        self.shown: list[dict] = []
        self.updated: list[dict] = []

    def show_econ_brief(self, view):
        self.shown.append(view)

    def update_econ_brief(self, view):
        self.updated.append(view)


def _host(tmp_path, *, mode="DESK", enabled=True, morning=True):
    from ui.services.trade_mentor_service import TradeMentorService

    mentor = TradeMentorService(
        idle_seconds=lambda: 0.0,
        session_locked=lambda: False,
        state_path=tmp_path / "trade_mentor_slots.json",
    )
    mentor.enabled = lambda: enabled
    review = _Review()
    host = SimpleNamespace(
        trading_panel=SimpleNamespace(alert_center=SimpleNamespace(chart_review=review)),
        trade_mentor_service=mentor,
        econ_reminder_service=SimpleNamespace(morning_has_started=lambda: morning),
        _auto_mode_now=lambda: mode,
    )
    return host, review


def _on_view(host, view):
    from ui.app import MainWindow

    MainWindow._on_econ_view(host, view)


def test_the_block_pops_once_per_session_and_not_again_after_a_restart(tmp_path):
    host, review = _host(tmp_path)
    _on_view(host, _view())
    assert len(review.shown) == 1
    _on_view(host, _view())
    assert len(review.shown) == 1 and len(review.updated) == 1
    # A desk restart the same day: the state file remembers.
    again, review2 = _host(tmp_path)
    _on_view(again, _view())
    assert review2.shown == [] and len(review2.updated) == 1


def test_a_paste_after_the_block_was_shown_redraws_it(tmp_path):
    host, review = _host(tmp_path)
    _on_view(host, _view())
    fresh = _view(origin="today_brief", origin_text="from today's brief")
    _on_view(host, fresh)
    assert review.updated[-1]["origin"] == "today_brief"


def test_the_next_session_pops_again(tmp_path):
    host, review = _host(tmp_path)
    _on_view(host, _view())
    nxt = _view()
    nxt["session"] = "2026-09-28"
    _on_view(host, nxt)
    assert [view["session"] for view in review.shown] == [SESSION, "2026-09-28"]


@pytest.mark.parametrize("mode", ["AWAY", "EVENING"])
def test_nothing_pops_while_away(tmp_path, mode):
    host, review = _host(tmp_path, mode=mode)
    _on_view(host, _view())
    assert review.shown == []
    assert not host.trade_mentor_service.econ_brief_shown(SESSION)


def test_nothing_pops_when_the_mentor_is_off_or_before_the_morning(tmp_path):
    host, review = _host(tmp_path, enabled=False)
    _on_view(host, _view())
    early, review2 = _host(tmp_path, morning=False)
    _on_view(early, _view())
    assert review.shown == [] and review2.shown == []
