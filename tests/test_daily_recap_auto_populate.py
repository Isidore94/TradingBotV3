"""The Daily Recap fills itself in at 12:00 Pacific - trader request 2026-09-14.

Two halves. `daily_recap_schedule` is pure and is tested on fixed instants,
including the two DST regimes, so the noon read is noon on the trader's wall
clock in March and in November. The page tests inject a clock and a settings
reader, so nothing here waits for a real noon, and they prove the four things
the trader will notice: the timer is not running until the host starts it, the
page moves to TODAY and reads it once, an earlier tick does nothing, and a
Refresh after the close relabels today from "provisional" to a plain session.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, time, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import daily_recap_schedule as schedule  # noqa: E402 - path first, as the WS-DR tests do

PACIFIC = schedule.PACIFIC
MONDAY = date(2026, 9, 14)
SATURDAY = date(2026, 9, 12)
LABOR_DAY = date(2026, 9, 7)


def _at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=PACIFIC)


def test_the_calendar_still_says_what_these_tests_assume():
    import market_calendar

    assert market_calendar.is_session(MONDAY)
    assert not market_calendar.is_session(SATURDAY)
    assert not market_calendar.is_session(LABOR_DAY)


# ---------------------------------------------------------------------------
# the setting
# ---------------------------------------------------------------------------


def test_the_default_is_noon_pacific_in_the_traders_words():
    assert schedule.DEFAULT_AUTO_TIME == "12:00"
    assert schedule.parse_auto_time(schedule.DEFAULT_AUTO_TIME) == time(12, 0)


@pytest.mark.parametrize("raw", ["", "off", "OFF", "none", "never", None])
def test_an_empty_or_off_setting_disables_the_read(raw):
    assert schedule.parse_auto_time(raw) is None


@pytest.mark.parametrize("raw", ["noon", "25:00", "12:00-07:00", "12:60", "12pm"])
def test_a_mistyped_time_disables_rather_than_guesses(raw):
    assert schedule.parse_auto_time(raw) is None


def test_an_hour_alone_is_a_whole_hour():
    assert schedule.parse_auto_time("12") == time(12, 0)
    assert schedule.parse_auto_time(" 13:15 ") == time(13, 15)


def test_the_setting_is_read_through_the_desks_own_accessor():
    seen = {}

    def getter(key, default=None):
        seen["key"] = key
        seen["default"] = default
        return "14:30"

    assert schedule.auto_time_from_settings(getter) == time(14, 30)
    assert seen == {"key": schedule.SETTING_KEY, "default": schedule.DEFAULT_AUTO_TIME}


def test_an_unreadable_settings_file_means_not_due_never_a_raise():
    def getter(key, default=None):
        raise OSError("settings locked")

    assert schedule.auto_time_from_settings(getter) is None


# ---------------------------------------------------------------------------
# due or not
# ---------------------------------------------------------------------------


def test_not_due_a_minute_before_the_hour():
    assert (
        schedule.due_session(_at(MONDAY, 11, 59), auto_time=time(12, 0), last_fired_session=None)
        is None
    )


def test_due_on_the_hour_and_the_answer_is_todays_session():
    assert (
        schedule.due_session(_at(MONDAY, 12, 0), auto_time=time(12, 0), last_fired_session=None)
        == "2026-09-14"
    )


def test_a_desk_started_late_in_the_afternoon_still_reads_today():
    assert (
        schedule.due_session(_at(MONDAY, 15, 30), auto_time=time(12, 0), last_fired_session=None)
        == "2026-09-14"
    )


def test_once_fired_it_stays_fired_for_that_session():
    assert (
        schedule.due_session(
            _at(MONDAY, 12, 5), auto_time=time(12, 0), last_fired_session="2026-09-14"
        )
        is None
    )
    # And yesterday's fire does not block today.
    assert (
        schedule.due_session(
            _at(MONDAY, 12, 5), auto_time=time(12, 0), last_fired_session="2026-09-11"
        )
        == "2026-09-14"
    )


@pytest.mark.parametrize("day", [SATURDAY, LABOR_DAY])
def test_a_non_session_day_is_never_due(day):
    assert (
        schedule.due_session(_at(day, 12, 0), auto_time=time(12, 0), last_fired_session=None)
        is None
    )


def test_a_disabled_setting_is_never_due():
    assert (
        schedule.due_session(_at(MONDAY, 12, 0), auto_time=None, last_fired_session=None)
        is None
    )


def test_noon_is_noon_on_the_wall_clock_in_both_dst_regimes():
    """19:00 UTC is noon Pacific in March (PDT) and 11:00 in November (PST)."""
    march = date(2026, 3, 10)
    november = date(2026, 11, 10)
    assert (
        schedule.due_session(
            datetime(2026, 3, 10, 19, 0, tzinfo=timezone.utc),
            auto_time=time(12, 0),
            last_fired_session=None,
        )
        == march.isoformat()
    )
    assert (
        schedule.due_session(
            datetime(2026, 11, 10, 19, 0, tzinfo=timezone.utc),
            auto_time=time(12, 0),
            last_fired_session=None,
        )
        is None
    )
    assert (
        schedule.due_session(
            datetime(2026, 11, 10, 20, 0, tzinfo=timezone.utc),
            auto_time=time(12, 0),
            last_fired_session=None,
        )
        == november.isoformat()
    )


def test_a_naive_clock_is_converted_never_relabelled():
    naive = datetime(2026, 9, 14, 12, 30)
    assert schedule.due_session(
        naive, auto_time=time(12, 0), last_fired_session=None
    ) == schedule.due_session(
        naive.astimezone(), auto_time=time(12, 0), last_fired_session=None
    )


def test_a_calendar_refusal_asks_nothing(monkeypatch):
    import market_calendar

    def refuse(_day):
        raise market_calendar.SessionCalendarError("outside the validated range")

    monkeypatch.setattr(market_calendar, "is_session", refuse)
    assert (
        schedule.due_session(_at(MONDAY, 12, 0), auto_time=time(12, 0), last_fired_session=None)
        is None
    )


def test_the_next_fire_skips_the_weekend():
    friday_afternoon = _at(date(2026, 9, 11), 13, 0)
    assert schedule.next_fire_at(friday_afternoon, auto_time=time(12, 0)) == _at(MONDAY, 12, 0)
    assert schedule.next_fire_at(_at(MONDAY, 9, 0), auto_time=time(12, 0)) == _at(MONDAY, 12, 0)
    assert schedule.next_fire_at(_at(MONDAY, 9, 0), auto_time=None) is None


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _process(qapp, rounds: int = 4) -> None:
    for _ in range(rounds):
        qapp.processEvents()


@pytest.fixture
def make_page(qapp):
    built = []

    def _build(*, clock, auto_time="12:00"):
        from ui.panels.daily_recap_panel import DailyRecapPanel

        page = DailyRecapPanel(clock=clock, auto_time_reader=lambda: auto_time)
        # The read itself is a worker over the live stores; these tests are
        # about WHEN it is asked for, so the ask is recorded rather than run.
        page.reads = []
        page.reload = lambda: page.reads.append(page.session_date())
        built.append(page)
        page.resize(1640, 980)
        page.show()
        _process(qapp)
        return page

    yield _build

    for page in built:
        try:
            page.shutdown()
        except Exception:
            pass
        page.hide()
        page.deleteLater()
    _process(qapp)


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.mark.qt
def test_the_timer_waits_for_the_host_and_stops_on_shutdown(make_page):
    page = make_page(clock=_Clock(_at(MONDAY, 9, 0)))

    assert not page._auto_timer.isActive(), "a timer started in a constructor"
    page.start()
    assert page._auto_timer.isActive()
    assert page._auto_timer.interval() == 60_000
    page.shutdown()
    assert not page._auto_timer.isActive()


@pytest.mark.qt
def test_before_noon_the_page_keeps_the_last_completed_session(make_page):
    page = make_page(clock=_Clock(_at(MONDAY, 11, 59)))

    assert page.poll_auto_read() is None
    assert page.session_date() == "2026-09-11"
    assert page.reads == []
    assert page.auto_fired_session() is None


@pytest.mark.qt
def test_at_noon_the_page_moves_to_today_and_reads_it_once(make_page):
    clock = _Clock(_at(MONDAY, 12, 0))
    page = make_page(clock=clock)
    assert page.session_date() == "2026-09-11", "the page opens on the completed session"

    assert page.poll_auto_read() == "2026-09-14"
    assert page.session_date() == "2026-09-14"
    assert page.reads == ["2026-09-14"]
    assert "provisional" in page.session_picker.currentText(), (
        "noon Pacific is an hour before the close; the entry must say so"
    )

    clock.now = _at(MONDAY, 12, 1)
    assert page.poll_auto_read() is None
    assert page.reads == ["2026-09-14"], "one automatic read per session"
    assert page.auto_fired_session() == "2026-09-14"


@pytest.mark.qt
def test_a_desk_started_after_noon_reads_today_on_its_first_tick(make_page):
    page = make_page(clock=_Clock(_at(MONDAY, 15, 45)))

    assert page.poll_auto_read() == "2026-09-14"
    assert page.session_date() == "2026-09-14"
    assert page.reads == ["2026-09-14"]


@pytest.mark.qt
def test_the_traders_own_time_setting_is_honoured(make_page):
    page = make_page(clock=_Clock(_at(MONDAY, 12, 0)), auto_time="13:15")

    assert page.poll_auto_read() is None
    page._clock = _Clock(_at(MONDAY, 13, 15))
    assert page.poll_auto_read() == "2026-09-14"


@pytest.mark.qt
def test_an_off_setting_never_moves_the_page(make_page):
    page = make_page(clock=_Clock(_at(MONDAY, 12, 0)), auto_time="off")

    assert page.poll_auto_read() is None
    assert page.session_date() == "2026-09-11"
    assert page.next_auto_read_at() is None


@pytest.mark.qt
def test_a_saturday_noon_reads_nothing(make_page):
    page = make_page(clock=_Clock(_at(SATURDAY, 12, 0)))

    assert page.poll_auto_read() is None
    assert page.reads == []


@pytest.mark.qt
def test_after_the_close_a_refresh_relabels_today_as_a_completed_session(make_page, qapp):
    clock = _Clock(_at(MONDAY, 12, 0))
    page = make_page(clock=clock)
    page.poll_auto_read()
    assert "provisional" in page.session_picker.currentText()

    # 13:05 Pacific is after the 16:00 ET close: today is now a completed
    # session and the list must say so on the next read, with the same
    # session still selected.
    clock.now = _at(MONDAY, 13, 5)
    page._refresh_session_picker()
    assert page.session_picker.itemData(0) == "2026-09-14"
    assert page.session_date() == "2026-09-14"
    assert "provisional" not in page.session_picker.currentText()
    assert not any(
        "provisional" in page.session_picker.itemText(i)
        for i in range(page.session_picker.count())
    )


@pytest.mark.qt
def test_a_refresh_before_the_close_keeps_the_list_and_the_selection(make_page):
    clock = _Clock(_at(MONDAY, 10, 0))
    page = make_page(clock=clock)
    page.session_picker.setCurrentIndex(2)
    chosen = page.session_date()
    entries = [page.session_picker.itemText(i) for i in range(page.session_picker.count())]

    page._refresh_session_picker()
    assert page.session_date() == chosen
    assert [
        page.session_picker.itemText(i) for i in range(page.session_picker.count())
    ] == entries


@pytest.mark.qt
def test_the_next_read_is_reported_on_the_traders_wall_clock(make_page):
    page = make_page(clock=_Clock(_at(date(2026, 9, 11), 13, 0)))

    assert page.next_auto_read_at() == _at(MONDAY, 12, 0)


def test_the_desk_starts_the_timer_after_the_window_shows():
    """A source-level pin: the host starts it in `showEvent`, beside the
    Trade Mentor's start, and never in the constructor."""
    source = (ROOT / "scripts" / "ui" / "app.py").read_text(encoding="utf-8")
    show_event = source[source.index("def showEvent(self, event)") :]
    show_event = show_event[: show_event.index("\n    def ")]
    assert "self.daily_recap_panel.start()" in show_event
    constructor = source[source.index("def __init__(") : source.index("def showEvent(")]
    assert "daily_recap_panel.start()" not in constructor


def test_the_selftest_can_reach_the_schedule_module():
    """`daily_recap_schedule` is a top-level import of the panel, which
    `ui.app` imports at module level, so a frozen bundle collects it without a
    selftest entry - the same reason the panel itself has none."""
    import importlib

    import ui.panels.daily_recap_panel as panel

    assert importlib.import_module("daily_recap_schedule") is panel.daily_recap_schedule
