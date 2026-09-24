"""Econ warnings: 07:00 PT (10:00 ET) cutoff, T-30 and T-0, no re-fire after restart,
late drop, and routing by Auto mode (DESK/OFF on the desk, AWAY/EVENING phone).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

ET = ZoneInfo("America/New_York")
SESSION = "2026-09-24"


def _at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 9, 24, hour, minute, second, tzinfo=ET)


def _view(rows=None):
    return {
        "session": SESSION,
        "origin": "today_brief",
        "summary_lines": [],
        "today": rows
        if rows is not None
        else [
            {"date": SESSION, "time_et": "06:00", "label": "Early release"},
            {"date": SESSION, "time_et": "08:30", "label": "Jobless claims"},
            {"date": SESSION, "time_et": "10:00", "label": "New home sales"},
            {"date": SESSION, "time_et": "10:00", "label": "Michigan sentiment"},
            {"date": SESSION, "time_et": "13:00", "label": "7-year Treasury auction"},
            {"date": SESSION, "time_et": "", "label": "Costco earnings, after the close"},
        ],
        "week": [],
    }


@pytest.fixture()
def make_service(tmp_path):
    from ui.services.econ_reminder_service import EconReminderService

    state = tmp_path / "econ_fired.json"
    sent: list[tuple[str, str]] = []

    def _push(title, message, **_kwargs):
        sent.append((title, message))
        return {"ok": True}

    built = []

    def _make(*, mode="DESK", engine_enabled=True, now=_at(9, 0)):
        service = EconReminderService(
            engine_enabled=engine_enabled,
            clock=lambda: now,
            loader=lambda _session: _view(),
            push=_push,
            state_path=state,
            mode=mode,
        )
        shown: list[dict] = []
        service.reminderFired.connect(shown.append)
        service.shown = shown
        service.sent = sent
        built.append(service)
        return service

    yield _make
    for service in built:
        service.shutdown()


def test_only_timed_events_at_or_after_seven_pacific_are_planned(make_service):
    """06:00 and 08:30 ET are before 07:00 PT: listed in the block, never warned."""
    service = make_service()
    service.apply_view(_view())
    keys = [item["key"] for item in service.planned()]
    assert keys == [
        f"{SESSION}|10:00|t30", f"{SESSION}|10:00|t0",
        f"{SESSION}|13:00|t30", f"{SESSION}|13:00|t0",
    ]


def test_events_at_one_time_share_one_warning(make_service):
    service = make_service()
    service.apply_view(_view())
    (soon,) = [item for item in service.planned() if item["key"] == f"{SESSION}|10:00|t30"]
    assert soon["message"].startswith("In 30 min: New home sales; Michigan sentiment")
    assert "10:00 a.m. ET / 7:00 a.m. PT" in soon["message"]
    assert soon["at"] == _at(9, 30)


def test_t30_then_t0_fire_once_each(make_service):
    service = make_service()
    service.apply_view(_view())
    assert service.tick(_at(9, 29, 59)) == []
    fired = service.tick(_at(9, 30, 10))
    assert [item["stage"] for item in fired] == ["t30"]
    assert fired[0]["message"].startswith("In 30 min:")
    assert service.tick(_at(9, 31)) == []
    fired = service.tick(_at(10, 0, 5))
    assert [item["stage"] for item in fired] == ["t0"]
    assert fired[0]["message"].startswith("Now: New home sales")
    assert [item["stage"] for item in service.shown] == ["t30", "t0"]


def test_a_restart_never_re_fires(make_service):
    first = make_service()
    first.apply_view(_view())
    assert first.tick(_at(9, 30, 5))
    second = make_service()
    second.apply_view(_view())
    assert second.tick(_at(9, 31)) == []
    assert [item["stage"] for item in second.tick(_at(10, 0, 1))] == ["t0"]


def test_a_warning_more_than_two_minutes_late_is_dropped(make_service):
    service = make_service()
    service.apply_view(_view())
    # The desk opened at 09:33: the 09:30 warning is dropped, not sent late.
    assert service.tick(_at(9, 33)) == []
    assert service.fired()[f"{SESSION}|10:00|t30"] == "dropped"
    assert f"{SESSION}|08:30|t0" not in service.fired()
    assert service.shown == []


def test_desk_mode_shows_on_the_desk_and_never_pushes(make_service):
    service = make_service(mode="DESK")
    service.apply_view(_view())
    fired = service.tick(_at(9, 30, 1))
    service.wait_for_pushes()
    assert fired and fired[0]["phone"] is False
    assert service.sent == []
    assert service.shown


def test_off_mode_is_at_the_desk_too(make_service):
    service = make_service(mode="OFF")
    service.apply_view(_view())
    service.tick(_at(9, 30, 1))
    service.wait_for_pushes()
    assert service.sent == []


@pytest.mark.parametrize("mode", ["AWAY", "EVENING"])
def test_away_and_evening_push_to_the_phone(make_service, mode):
    service = make_service(mode=mode)
    service.apply_view(_view())
    service.tick(_at(9, 30, 1))
    service.wait_for_pushes()
    assert service.sent and service.sent[0][1].startswith("In 30 min: New home sales")


def test_a_mode_flip_takes_effect_at_once(make_service):
    service = make_service(mode="DESK")
    service.apply_view(_view())
    service.on_auto_mode_changed("DESK", "AWAY")
    service.tick(_at(9, 30, 1))
    service.wait_for_pushes()
    assert service.sent


def test_only_the_engine_machine_pushes(make_service):
    service = make_service(mode="AWAY", engine_enabled=False)
    service.apply_view(_view())
    fired = service.tick(_at(9, 30, 1))
    service.wait_for_pushes()
    assert fired and service.sent == []


def test_a_new_view_reschedules(make_service):
    service = make_service()
    service.apply_view(_view())
    service.apply_view(_view([{"date": SESSION, "time_et": "11:00", "label": "Fed speaker"}]))
    assert [item["key"] for item in service.planned()] == [
        f"{SESSION}|11:00|t30", f"{SESSION}|11:00|t0",
    ]


def test_a_pasted_brief_triggers_a_refresh(make_service, monkeypatch):
    service = make_service()
    calls = []
    monkeypatch.setattr(service, "refresh", lambda: calls.append(1))
    service.on_journal_entry({"origin": "desk_tab"})
    service.on_journal_entry({"origin": "external_forecast"})
    assert calls == [1]


def test_the_view_loads_off_the_qt_thread_and_arrives_as_a_signal(make_service):
    import threading
    import time

    seen_threads = []

    def _loader(_session):
        seen_threads.append(threading.current_thread() is threading.main_thread())
        return _view()

    service = make_service()
    service._loader = _loader
    views = []
    service.viewChanged.connect(views.append)
    service.refresh()
    deadline = time.monotonic() + 5
    while not views and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)
    assert views and views[0]["session"] == SESSION
    assert seen_threads == [False]
