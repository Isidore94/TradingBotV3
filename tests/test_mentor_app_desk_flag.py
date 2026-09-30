"""P1: `mentor_app_enabled` on = the desk builds no Trade Mentor; off = today's desk.

With the flag on the desk never writes `MENTOR_ASKED`, never touches
`trade_mentor_slots.json`, and the Alert Center shows "Open Trade Mentor" instead of
the popup. The status-bar button badges due, unanswered slots from the app's file.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj14b_desk_isolation import fresh_mentor_pull_tally  # noqa: E402,F401
from tj14b_support import SESSION, pacific, slot_at  # noqa: E402

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the desk is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402


def _flag(monkeypatch, value):
    import project_paths

    real = project_paths.get_local_setting

    def get(key, default=None):
        if key == "mentor_app_enabled":
            return value
        return real(key, default)

    monkeypatch.setattr(project_paths, "get_local_setting", get)


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, False), (False, False), ("", False), ("off", False), (True, True), ("true", True), (1, True)]
)
def test_the_flag_is_off_unless_explicitly_on(monkeypatch, raw, expected):
    from ui.services.mentor_launcher import mentor_app_enabled

    _flag(monkeypatch, raw)
    assert mentor_app_enabled() is expected


@pytest.fixture()
def desk_on(monkeypatch):
    import journal_store
    from project_paths import TRADE_MENTOR_SLOTS_FILE
    from ui.app import MainWindow
    from ui.state import UiState

    QApplication.instance() or QApplication([])
    _flag(monkeypatch, True)
    slots = Path(TRADE_MENTOR_SLOTS_FILE)
    slots.parent.mkdir(parents=True, exist_ok=True)
    slots.write_text(json.dumps({"slots": {}}), encoding="utf-8")
    before = slots.stat().st_mtime_ns
    events: list = []
    real = journal_store.JournalStore.record_opportunity_event

    def spy(self, **kwargs):
        events.append(kwargs.get("event_type"))
        return real(self, **kwargs)

    monkeypatch.setattr(journal_store.JournalStore, "record_opportunity_event", spy)
    launched: list = []
    import ui.services.mentor_launcher as launcher

    monkeypatch.setattr(launcher, "launch_or_focus", lambda **kwargs: launched.append(1))
    window = MainWindow(UiState(workspace_mode="workspace"))

    QApplication.processEvents()
    try:
        yield window, slots, before, events, launched
    finally:
        window.close()


def test_with_the_flag_on_the_desk_builds_no_mentor_and_writes_nothing(desk_on):
    window, slots, before, events, _launched = desk_on
    review = window.trading_panel.alert_center.chart_review
    assert window.mentor_app_enabled is True
    assert window.trade_mentor_service is None and window.trade_mentor_context_service is None
    assert review.mentor_card is None and review.mentor_popup is None
    # Every desk path that used to reach the card is inert.
    window._sync_trade_mentor_label()
    window.econ_reminder_service.viewChanged.emit({"session": SESSION.isoformat()})
    QApplication.processEvents()
    assert "MENTOR_ASKED" not in events
    assert slots.stat().st_mtime_ns == before, "the desk never touches the app's slots file"


def test_the_alert_center_offers_open_trade_mentor_instead_of_the_popup(desk_on):
    window, _slots, _before, _events, launched = desk_on
    review = window.trading_panel.alert_center.chart_review
    button = review.open_trade_mentor_button
    assert button.text() == "Open Trade Mentor"
    assert review.isAncestorOf(button), "it sits in the verb row where Give a read was"
    button.click()
    window._open_mentor_on_missing_input(object())
    assert launched == [1, 1], "both the button and the Inputs chip open the app"


def test_the_desk_button_badges_due_unanswered_slots(desk_on):
    window, slots, _before, _events, _launched = desk_on
    slot = slot_at(SESSION, 9)
    payload = {"slots": {slot.slot_id: {"delivered_at": pacific(SESSION, 9).isoformat(), "answered_at": "", "skipped_reason": ""}}}
    slots.write_text(json.dumps(payload), encoding="utf-8")
    window._mentor_badge._checked = float("-inf")
    import ui.app as app_module

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return pacific(SESSION, 9, 20)

    original = app_module.datetime
    app_module.datetime = _Clock
    try:
        assert window._refresh_mentor_badge() == 1
        assert window.trade_mentor_app_button.text() == "Trade Mentor (1)"
        payload["slots"][slot.slot_id]["answered_at"] = pacific(SESSION, 9, 25).isoformat()
        slots.write_text(json.dumps(payload), encoding="utf-8")
        window._mentor_badge._checked = float("-inf")
        assert window._refresh_mentor_badge() == 0
        assert window.trade_mentor_app_button.text() == "Trade Mentor"
    finally:
        app_module.datetime = original


def test_the_badge_reader_is_read_only_and_restats_at_most_once_a_second(tmp_path):
    from ui.services.mentor_badge import MentorSlotsBadge

    path = tmp_path / "trade_mentor_slots.json"
    slot = slot_at(SESSION, 10)
    path.write_text(
        json.dumps({"slots": {slot.slot_id: {"delivered_at": "x", "answered_at": "", "skipped_reason": ""}}}),
        encoding="utf-8",
    )
    ticks = {"t": 100.0}
    badge = MentorSlotsBadge(path, clock=lambda: ticks["t"])
    inside = pacific(SESSION, 10, 30)
    assert badge.count(inside) == 1
    path.write_text(json.dumps({"slots": {}}), encoding="utf-8")
    ticks["t"] += 0.5
    assert badge.count(inside) == 1, "no re-stat inside a second"
    ticks["t"] += 1.0
    assert badge.count(inside) == 0
    assert badge.count(inside + timedelta(hours=2)) == 0
