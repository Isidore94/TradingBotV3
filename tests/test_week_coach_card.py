"""Week Review coach section: renders only, reads and writes on a worker."""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QLabel, QToolButton  # noqa: E402

_app = QApplication.instance() or QApplication([])
MAIN = threading.get_ident()


def _wait(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    _app.processEvents()
    assert condition()


VIEW = {
    "week": "2026-W38", "month": "", "recorded": True, "covered_weeks": ["2026-W38"],
    "sessions": ["2026-09-14", "2026-09-15"], "trades_n": 15, "thin_rows": 4,
    "edge": [], "leaks": [{"group": "by_time_of_day", "label": "Time of day", "key": "close", "n": 10,
                           "pnl_known_n": 10, "r_n": 10, "wins": 0, "losses": 10, "avg_r": -0.5,
                           "pnl_cad": -200.0, "metric": "avg_r", "value": -0.5, "tier": 0,
                           "id": "week:2026-W38:by_time_of_day:close"}],
    "repeats": {"lessons": [{"part": "stop", "text": "chasing", "n": 2}], "rules": []},
    "rule_kept": {"n": 2, "rate": 0.5}, "calls": {"n": 2, "rate": 0.5},
    "trend": [{"week": "2026-W37", "recorded": False},
              {"week": "2026-W38", "recorded": True, "pnl_cad": -50.0, "pnl_known_n": 15,
               "rule_kept_rate": 0.5, "rule_checks_n": 2, "calls": {"rate": 0.5, "n": 2}}],
    "questions": [
        {"id": "q-1", "text": "Do my afternoon trades lose?", "status": "answered", "answer": {
            "claims": [{"text": "Yes.", "flag": "", "citations": [
                {"id": "week:2026-W38:by_time_of_day:close", "session": ""},
                {"id": "trade:pm-1-0", "session": "2026-09-15"}]}],
            "dropped_n": 1, "model": "fake", "small_sample": False}},
        {"id": "q-2", "text": "Do I keep my rules?", "status": "pending", "answer": None},
    ],
}


class _Reader:
    def __init__(self):
        self.calls: list[tuple[str, bool]] = []
        self.threads: list[int] = []

    def __call__(self, week, *, month=False):
        self.calls.append((week, month))
        self.threads.append(threading.get_ident())
        return {**VIEW, "week": week or VIEW["week"]}


@pytest.fixture
def card():
    from ui.widgets.week_coach_card import WeekCoachCard

    reader = _Reader()
    asked: list[tuple[str, str, int]] = []

    def ask(text, *, week=""):
        asked.append((text, week, threading.get_ident()))
        return {"id": "q-3"}

    widget = WeekCoachCard(read=reader, ask=ask)
    widget.reader, widget.asked = reader, asked
    yield widget
    widget.shutdown()
    widget.deleteLater()


def test_the_card_reads_off_the_qt_thread_and_shows_n_everywhere(card):
    card.set_week("2026-W38")
    assert card.reader.calls == []  # choosing a week reads nothing
    card.load()
    _wait(lambda: card.reader.calls and "2026-W38" in card.week_label.text())
    assert card.reader.threads and all(ident != MAIN for ident in card.reader.threads)
    assert "(n 10)" in card.leaks_label.text()
    assert "too few to tell" in card.edge_label.text()
    assert "x2" in card.repeats_label.text()
    assert card.trend.item(0, 1).text() == "no record"
    assert card.trend.item(1, 1).text().endswith("(n 15)")
    texts = " ".join(label.text() for label in card.questions.findChildren(QLabel))
    assert "answered tonight" in texts  # the pending question
    assert "uncited — not shown" in texts


def test_a_session_citation_opens_that_day_and_a_week_row_does_not(card):
    card.render(VIEW)
    opened: list[str] = []
    card.openSessionRequested.connect(opened.append)
    buttons = {button.text(): button for button in card.questions.findChildren(QToolButton)}
    assert not buttons["week:2026-W38:by_time_of_day:close"].isEnabled()
    buttons["trade:pm-1-0"].click()
    assert opened == ["2026-09-15"]


def test_the_arrows_and_month_toggle_reread_on_the_worker(card):
    card.load("2026-W38")
    _wait(lambda: len(card.reader.calls) == 1 and card.week_label.text() == "2026-W38")
    card.prev_button.click()
    _wait(lambda: card.reader.calls[-1] == ("2026-W37", False) and card.week_label.text() == "2026-W37")
    card.next_button.click()
    _wait(lambda: card.reader.calls[-1] == ("2026-W38", False) and card.week_label.text() == "2026-W38")
    card.month_button.click()
    _wait(lambda: card.reader.calls[-1] == ("2026-W38", True))
    card.next_button.click()  # a whole month on in Month view
    _wait(lambda: card.reader.calls[-1] == ("2026-W40", True))
    assert all(ident != MAIN for ident in card.reader.threads)


def test_asking_saves_on_the_worker_and_says_answered_tonight(card):
    card.render(VIEW)
    card.ask_box.setText("Do my afternoon trades lose?")
    card.ask_button.click()
    _wait(lambda: card.asked and card.reader.calls)
    text, week, ident = card.asked[0]
    assert text == "Do my afternoon trades lose?" and week == "2026-W38" and ident != MAIN
    assert "Answered tonight" in card.ask_note.text()
    assert card.ask_box.text() == ""


def test_week_review_page_carries_the_coach_and_routes_its_days(tmp_path, monkeypatch):
    import week_coach
    from ui.panels.weekend_prep_panel import WeekReviewPage
    from ui.services.weekend_prep_service import WeekendPrepService

    reader = _Reader()
    monkeypatch.setattr(week_coach, "read_view", reader)
    service = WeekendPrepService(state_path=tmp_path / "state.json", now=datetime(2026, 9, 19, 10, 0))
    page = WeekReviewPage(service)
    try:
        assert reader.calls == []  # building the page reads nothing
        assert page.coach._week == "2026-W38"
        opened: list[str] = []
        page.openSessionRequested.connect(opened.append)
        page.coach.openSessionRequested.emit("2026-09-15")
        assert opened == ["2026-09-15"]
        page.reload()
        _wait(lambda: reader.calls and not page._reading)
        assert reader.calls[0] == ("2026-W38", False)
        assert all(ident != MAIN for ident in reader.threads)
    finally:
        page.shutdown()
        service.shutdown()
        page.deleteLater()
