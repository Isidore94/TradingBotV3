"""TJ-13A items 1 and 2: the night window applies seven days a week.

Trader, 2026-09-19: *"I keep the computer on overnight in the weekends too. … I
always want the bot to run overnight never during the day so I can restart it or
use it for market prep."* (decision 0021 answer 19, plan.md §12.4 TJ-13 items 5
and 6.)

Two things are pinned here and they are different questions:

1. **The clock.** ``is_weekend`` used to open the configured window for the whole
   of Saturday and Sunday, so a firing at 14:00 Pacific on a Saturday was inside
   the window. The trader uses the desk by day at the weekend; the night window
   now applies every day. The market-session block is a SEPARATE, invariant gate
   and nothing here touches it.
2. **Which night it is.** ``night_kind`` names a night on the exchange calendar,
   so the weekend slate follows the sessions rather than the weekday numbers: a
   Monday holiday moves the Sunday slate to Monday night, and a Friday holiday
   starts the weekend a night early.

Times are written in Pacific because that is the clock the trader and the
scheduled task keep (``TradingBotV3 AI Jobs`` fires 22:00 local, every 30 minutes
for 8 hours, ``DaysInterval=1`` - measured 2026-09-19). The window itself is
stored in ET and the live setting is 01:00-09:00 ET, which is that same night.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from zoneinfo import ZoneInfo  # noqa: E402

ET = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")

#: The live bounds, read from the desk's settings on 2026-09-19.
LIVE_START = "01:00"
LIVE_END = "09:00"


def _settings(**values):
    """Patch the settings reader the window module actually resolves through.

    ``window._paths()`` reaches ``scripts.project_paths`` (a namespace-package
    import), which is a different module object from a plain
    ``import project_paths`` - the precedent is
    ``tests/test_ai_jobs_store_window.py``.
    """
    from ai_jobs import store

    return mock.patch.object(
        store._paths(),
        "get_local_setting",
        lambda key, default=None: values.get(key, default),
    )


def _no_session_block(monkeypatch):
    """Neutralise the market-session gate so the WINDOW is what is measured.

    Patches the block itself, never the calendar: an unanswerable calendar
    blocks, so stubbing ``_session_bounds`` would not mean "no session today".
    """
    from ai_jobs import window

    monkeypatch.setattr(window, "market_session_block", lambda now=None: "")


# ---------------------------------------------------------------------------
# item 1 - the clock
# ---------------------------------------------------------------------------


def test_a_saturday_afternoon_is_outside_the_night_window(monkeypatch):
    """14:00 Pacific on a Saturday is the trader's desk time, not the bot's."""
    from ai_jobs import window

    _no_session_block(monkeypatch)
    saturday_afternoon = datetime(2026, 9, 19, 14, 0, tzinfo=PACIFIC)
    assert saturday_afternoon.weekday() == 5

    with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
        assert window.in_offhours_window(saturday_afternoon) is False
        allowed, reason = window.launch_allowed(saturday_afternoon)
    assert allowed is False
    assert "window" in reason


def test_a_sunday_afternoon_is_outside_the_night_window(monkeypatch):
    """The same rule on the other weekend day: nothing runs by day, any day."""
    from ai_jobs import window

    _no_session_block(monkeypatch)
    sunday_afternoon = datetime(2026, 9, 20, 13, 0, tzinfo=PACIFIC)
    assert sunday_afternoon.weekday() == 6

    with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
        assert window.in_offhours_window(sunday_afternoon) is False


def test_a_saturday_night_is_inside_the_night_window(monkeypatch):
    """The companion guard: removing the weekend exemption must not close the
    weekend NIGHT, which is the one the heavy slate runs on."""
    from ai_jobs import window

    _no_session_block(monkeypatch)
    with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
        # 23:00 Pacific Saturday == 02:00 ET Sunday: inside 01:00-09:00 ET.
        assert window.in_offhours_window(datetime(2026, 9, 19, 23, 0, tzinfo=PACIFIC))
        # 22:00 Pacific Saturday == 01:00 ET Sunday: the moment it opens.
        assert window.in_offhours_window(datetime(2026, 9, 19, 22, 0, tzinfo=PACIFIC))
        # 05:59 Pacific Sunday == 08:59 ET: still open.
        assert window.in_offhours_window(datetime(2026, 9, 20, 5, 59, tzinfo=PACIFIC))
        # 06:00 Pacific Sunday == 09:00 ET: closed.
        assert not window.in_offhours_window(datetime(2026, 9, 20, 6, 0, tzinfo=PACIFIC))


def test_a_weekend_window_closes_that_morning_and_not_on_monday(monkeypatch):
    """The number, not the shape.

    The weekend branch of ``window_close_at`` walked forward to the next weekday
    morning, so a job starting at 23:00 Pacific on a Saturday was told it had
    until 09:00 ET **Monday** - 31 hours - and every ``reserve_minutes`` check
    passed on a number that was never true. The real answer is 09:00 ET Sunday,
    which is 06:00 Pacific Sunday: seven hours, 420 minutes.
    """
    from ai_jobs import window

    _no_session_block(monkeypatch)
    saturday_night = datetime(2026, 9, 19, 23, 0, tzinfo=PACIFIC)

    with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
        close = window.window_close_at(saturday_night)
        remaining = window.minutes_until_window_close(saturday_night)

    assert close is not None
    assert close.astimezone(PACIFIC) == datetime(2026, 9, 20, 6, 0, tzinfo=PACIFIC)
    assert remaining == pytest.approx(420.0)


def test_the_market_session_block_is_still_a_separate_gate(monkeypatch):
    """Guard: item 1 is about the window only.

    The weekend short-circuit inside ``market_session_block`` is the sec 2 hard
    rule's own fail-closed design (it must not need the calendar on a day the
    exchange never opens), and this packet does not touch it.
    """
    from ai_jobs import window

    def _explode(day):  # pragma: no cover - must never be reached
        raise AssertionError("the session gate must not need the calendar on a weekend")

    monkeypatch.setattr(window, "_session_bounds", _explode)
    assert window.market_session_block(datetime(2026, 9, 19, 14, 0, tzinfo=PACIFIC)) == ""


# ---------------------------------------------------------------------------
# item 1 - --force buys the caps, never the clock
# ---------------------------------------------------------------------------


def _spy_slate(calls):
    """The REAL nightly slate with every job replaced by a recorder.

    ``dataclasses.replace`` keeps each slot's name, reserve, attempt cap and any
    field a later packet adds to ``JobSlot``, so this exercises the runner's own
    selection rather than a lookalike slot list.
    """
    from ai_jobs import runner

    def _record(name):
        def _run(**kwargs):
            calls.append(name)
            return {"reason": "ran"}

        return _run

    return [replace(slot, run=_record(slot.name)) for slot in runner.default_slots()]


def test_a_forced_daytime_run_starts_no_model_slot(tmp_path, monkeypatch):
    """``--force`` is for the attempt caps and the already-done check.

    It was also skipping the clock, which is how "run it now, I know it is 09:00
    ET on a Sunday" turned into a 14 GB model load in front of the trader's own
    market prep. The market-session block was never reachable by it; the WINDOW
    now is not either, for a slot that calls a model.
    """
    from ai_jobs import ledger, runner, store

    calls: list[str] = []
    slots = _spy_slate(calls)
    led = tmp_path / "ledger.jsonl"
    saturday_afternoon = datetime(2026, 9, 19, 14, 0, tzinfo=PACIFIC)

    with mock.patch.object(store, "store_available", return_value=(True, "ready")):
        with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
            report = runner.run_slots(
                slots,
                now=saturday_afternoon,
                force=True,
                only="ai_summary",
                ledger_path=led,
            )

    assert calls == [], "a forced daytime run must not start local inference"
    assert [row["status"] for row in report.results] == [ledger.STATUS_SKIPPED]


def test_a_forced_daytime_run_may_still_do_the_deterministic_work(tmp_path):
    """Guard on the other side of item 1.

    A deterministic slot costs seconds and calls no model, so forcing one by day
    stays exactly what it is today. Closing the window over these too would make
    ``--force`` useless for the repair it exists for.
    """
    from ai_jobs import runner, store

    calls: list[str] = []
    slots = _spy_slate(calls)
    led = tmp_path / "ledger.jsonl"
    saturday_afternoon = datetime(2026, 9, 19, 14, 0, tzinfo=PACIFIC)

    with mock.patch.object(store, "store_available", return_value=(True, "ready")):
        with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
            runner.run_slots(
                slots,
                now=saturday_afternoon,
                force=True,
                only="journal_import",
                ledger_path=led,
            )

    assert calls == ["journal_import"]


# ---------------------------------------------------------------------------
# item 2 - night kinds, on the exchange calendar
# ---------------------------------------------------------------------------

KINDS = ("weeknight", "saturday", "sunday")


def _night_kind(moment):
    from ai_jobs import runner

    return runner.night_kind(moment)


def test_night_kind_names_an_ordinary_weeks_three_kinds():
    """2026-09-14..09-21: sessions Mon-Fri, the weekend closed (verified against
    ``market_calendar.is_session`` on 2026-09-19)."""
    # Thursday night: a session behind it.
    assert _night_kind(datetime(2026, 9, 17, 23, 0, tzinfo=PACIFIC)) == "weeknight"
    # Friday night: still a weeknight - Friday WAS a session.
    assert _night_kind(datetime(2026, 9, 18, 23, 0, tzinfo=PACIFIC)) == "weeknight"
    # Saturday night: the first night with no session behind it.
    assert _night_kind(datetime(2026, 9, 19, 23, 0, tzinfo=PACIFIC)) == "saturday"
    # Sunday night: the last night before Monday's session.
    assert _night_kind(datetime(2026, 9, 20, 23, 0, tzinfo=PACIFIC)) == "sunday"


def test_a_night_keeps_its_kind_after_midnight():
    """A night is one night. The 02:00 Pacific firing of Saturday night belongs
    to Saturday's slate, not to Sunday's - and in ET it is already Sunday, which
    is exactly the seam a date-only reading gets wrong."""
    assert _night_kind(datetime(2026, 9, 20, 2, 0, tzinfo=PACIFIC)) == "saturday"
    assert _night_kind(datetime(2026, 9, 20, 5, 59, tzinfo=PACIFIC)) == "saturday"
    assert _night_kind(datetime(2026, 9, 21, 2, 0, tzinfo=PACIFIC)) == "sunday"


def test_a_monday_holiday_moves_the_sunday_slate_to_monday_night():
    """Labor Day 2026-09-07 is a Monday and not a session; 09-08 is.

    So Sunday night is NOT the last night before the next session and must not
    run the backlog slate; Monday night is.
    """
    assert _night_kind(datetime(2026, 9, 4, 23, 0, tzinfo=PACIFIC)) == "weeknight"
    assert _night_kind(datetime(2026, 9, 5, 23, 0, tzinfo=PACIFIC)) == "saturday"
    assert _night_kind(datetime(2026, 9, 6, 23, 0, tzinfo=PACIFIC)) != "sunday"
    assert _night_kind(datetime(2026, 9, 7, 23, 0, tzinfo=PACIFIC)) == "sunday"
    assert _night_kind(datetime(2026, 9, 8, 23, 0, tzinfo=PACIFIC)) == "weeknight"


def test_a_friday_holiday_starts_the_weekend_a_night_early():
    """Good Friday 2026-04-03 is not a session; Thursday 04-02 is, Monday 04-06 is.

    Friday night is then the first night with no session behind it, so the
    weekend slate starts there, and Sunday night is still the last one before a
    session.
    """
    assert _night_kind(datetime(2026, 4, 2, 23, 0, tzinfo=PACIFIC)) == "weeknight"
    assert _night_kind(datetime(2026, 4, 3, 23, 0, tzinfo=PACIFIC)) == "saturday"
    assert _night_kind(datetime(2026, 4, 5, 23, 0, tzinfo=PACIFIC)) == "sunday"
    assert _night_kind(datetime(2026, 4, 6, 23, 0, tzinfo=PACIFIC)) == "weeknight"


def test_every_night_of_a_holiday_week_has_exactly_one_kind():
    """A total function: no night falls through the three kinds.

    Walked over the Thanksgiving week, whose Thursday is closed and whose Friday
    is a half session - a shape no weekday rule gets right by accident.
    """
    day = date(2026, 11, 23)
    while day <= date(2026, 11, 30):
        moment = datetime(day.year, day.month, day.day, 23, 0, tzinfo=PACIFIC)
        kind = _night_kind(moment)
        assert kind in KINDS, f"{day} night returned {kind!r}"
        day += timedelta(days=1)
    # Thursday 11-26 is closed and Friday 11-27 is a (half) session, so Thursday
    # night has no session behind it and Friday night does.
    assert _night_kind(datetime(2026, 11, 27, 23, 0, tzinfo=PACIFIC)) == "weeknight"
    assert _night_kind(datetime(2026, 11, 28, 23, 0, tzinfo=PACIFIC)) == "saturday"
    assert _night_kind(datetime(2026, 11, 29, 23, 0, tzinfo=PACIFIC)) == "sunday"


def test_night_kind_decides_nothing_and_writes_nothing(tmp_path):
    """Pure: same moment, same answer, and no store is opened to get it."""
    moment = datetime(2026, 9, 19, 23, 0, tzinfo=PACIFIC)
    assert _night_kind(moment) == _night_kind(moment)
    assert sorted(p.name for p in tmp_path.iterdir()) == []
