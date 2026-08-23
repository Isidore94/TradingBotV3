"""R10.A / Sol blocker 1 - the after-close jobs run when they should, once each.

What Sol reproduced on 2026-08-24: at 13:10 PT the worker fired, the sweep
correctly deferred to close+35, and then `_learning_refresh_date` was stamped
anyway because the refresh had succeeded. At 13:40 nothing ran, because the day
was already marked done. **Autorun could never actually sweep.**

The fix is two jobs, two clocks and **two completion stamps**:

* the sweep runs no earlier than the real close + 35 minutes, once, and only
  when the switch is on;
* the refresh runs at close + grace, once - but not before the sweep whose rows
  it is supposed to read;
* a deferred or failed sweep leaves the day open and is retried;
* a successful refresh can never mark the sweep complete.

"The real close" means the real one. `market_calendar` models every close as
16:00 ET by design and every existing caller keeps that; the scheduler asks a
dedicated seam instead, so a half day is 13:00 ET here and nothing about
scanner or detector hours moves.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_early_close as ec  # noqa: E402

# Sol's day. Regular close 16:00 ET = 13:00 PT; sweep window opens 13:35 PT.
MONDAY = date(2026, 8, 24)
# NYSE closes 13:00 ET the day after Thanksgiving: 10:00 PT, sweep at 10:35 PT.
HALF_DAY = date(2026, 11, 27)
# Standard time either side of the DST change (2026-11-01).
POST_DST = date(2026, 11, 30)


class _Host:
    pass


def _host(*, enabled=True, sweep_done=None, refresh_done=None):
    from bounce_bot_lib.legacy import BounceBot

    host = _Host.__new__(_Host)
    host.SWEEP_AFTER_SCAN_WINDOW_MINUTES = BounceBot.SWEEP_AFTER_SCAN_WINDOW_MINUTES
    host._outcome_sweep_date = sweep_done
    host._learning_refresh_date = refresh_done
    host._sweep_autorun_enabled = lambda: enabled
    host.actual_session_close = BounceBot.actual_session_close.__get__(host, _Host)
    host._after_close_jobs_due = BounceBot._after_close_jobs_due.__get__(host, _Host)
    host._sweep_window_is_open = BounceBot._sweep_window_is_open.__get__(host, _Host)
    return host


def _at(day: date, hhmm: str) -> datetime:
    hour, minute = (int(part) for part in hhmm.split(":"))
    return datetime(day.year, day.month, day.day, hour, minute)


# ---------------------------------------------------------------------------
# the early-close seam
# ---------------------------------------------------------------------------
def test_the_day_after_thanksgiving_closes_early():
    assert ec.early_close_reason(HALF_DAY) == ec.REASON_DAY_AFTER_THANKSGIVING
    assert ec.session_close_time(HALF_DAY) == ec.EARLY_CLOSE


def test_thanksgiving_itself_is_not_a_session_and_has_no_early_close():
    assert ec.early_close_reason(date(2026, 11, 26)) is None


def test_christmas_eve_closes_early_when_it_is_a_session():
    assert ec.early_close_reason(date(2026, 12, 24)) == ec.REASON_CHRISTMAS_EVE
    # 2027: the 25th is a Saturday, so the 24th IS the holiday - not a session.
    assert ec.early_close_reason(date(2027, 12, 24)) is None


def test_july_third_closes_early_only_when_the_fourth_is_a_weekday():
    assert ec.early_close_reason(date(2025, 7, 3)) == ec.REASON_JULY_THIRD
    # 2026: 4 July is a Saturday, observed Friday the 3rd - closed all day.
    assert ec.early_close_reason(date(2026, 7, 3)) is None


def test_an_ordinary_session_closes_at_the_regular_time():
    assert ec.early_close_reason(MONDAY) is None
    assert ec.session_close_time(MONDAY) == ec.REGULAR_CLOSE


def test_a_weekend_has_no_close_to_be_early():
    assert ec.early_close_reason(date(2026, 8, 22)) is None


def test_a_date_outside_the_calendar_answers_regular_rather_than_raising():
    """Conservative in the only direction that matters: the sweep waits longer."""
    assert ec.early_close_reason(date(2099, 11, 27)) is None
    assert ec.session_close_time(date(2099, 11, 27)) == ec.REGULAR_CLOSE


def test_the_seam_says_how_it_decided():
    assert "day_after_thanksgiving" in ec.describe(HALF_DAY)
    assert "regular_close" in ec.describe(MONDAY)


def test_the_shared_calendar_is_untouched():
    """No detector or scanner hour moves because of this packet."""
    from market_calendar import REGULAR_CLOSE, session_close

    assert REGULAR_CLOSE == ec.REGULAR_CLOSE
    assert session_close(HALF_DAY).time() == REGULAR_CLOSE, (
        "market_calendar still models every close as regular, deliberately"
    )


# ---------------------------------------------------------------------------
# Sol's exact reproduction
# ---------------------------------------------------------------------------
def test_the_1310_tick_does_not_mark_the_day_done():
    """The bug: the refresh succeeded at 13:10 and the sweep was never retried."""
    host = _host()
    due = host._after_close_jobs_due(_at(MONDAY, "13:10"))
    assert due["sweep"] is False
    assert due["refresh"] is False, "the refresh waits for the sweep it must read"
    assert due["reason"] == "waiting_for_sweep"


def test_the_1340_tick_runs_both():
    due = _host()._after_close_jobs_due(_at(MONDAY, "13:40"))
    assert due["sweep"] is True and due["refresh"] is True


def test_the_sweep_opens_at_close_plus_thirty_five_not_before():
    host = _host()
    assert host._after_close_jobs_due(_at(MONDAY, "13:34"))["sweep"] is False
    assert host._after_close_jobs_due(_at(MONDAY, "13:35"))["sweep"] is True


def test_a_completed_sweep_is_not_repeated():
    host = _host(sweep_done=MONDAY)
    due = host._after_close_jobs_due(_at(MONDAY, "13:40"))
    assert due["sweep"] is False
    assert due["refresh"] is True, "the refresh may now run - the rows are there"


def test_a_completed_refresh_is_not_repeated():
    host = _host(sweep_done=MONDAY, refresh_done=MONDAY)
    due = host._after_close_jobs_due(_at(MONDAY, "13:40"))
    assert due == {"sweep": False, "refresh": False, "reason": "not_due", "today": MONDAY}


def test_a_successful_refresh_cannot_mark_the_sweep_complete():
    """The two stamps are separate variables; this pins that they stay separate."""
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    source = inspect.getsource(BounceBot._maybe_refresh_learning_after_close)
    sweep_stamp = source.index("self._outcome_sweep_date = today")
    refresh_stamp = source.index("self._learning_refresh_date = today")
    assert sweep_stamp < refresh_stamp
    # the sweep stamp is inside the branch that ran the sweep, after a deferral check
    assert 'if swept.get("deferred")' in source
    assert source.index('if swept.get("deferred")') < sweep_stamp


def test_a_deferred_sweep_is_retried_rather_than_stamped():
    """A deferral must leave the day open."""
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    source = inspect.getsource(BounceBot._maybe_refresh_learning_after_close)
    assert "it will be retried" in source
    # ...and the stamp sits in the else branch, not beside the deferral log.
    deferred_at = source.index("Outcome sweep still deferred")
    stamp_at = source.index("self._outcome_sweep_date = today")
    assert deferred_at < stamp_at


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------
def test_the_switch_off_means_the_sweep_never_runs_automatically():
    host = _host(enabled=False)
    for moment in ("13:10", "13:40", "18:00"):
        assert host._after_close_jobs_due(_at(MONDAY, moment))["sweep"] is False


def test_the_switch_off_lets_the_refresh_run_at_close_plus_grace():
    """Nothing to wait for, so the old timing stands."""
    due = _host(enabled=False)._after_close_jobs_due(_at(MONDAY, "13:10"))
    assert due["refresh"] is True


# ---------------------------------------------------------------------------
# calendar shapes
# ---------------------------------------------------------------------------
def _sweep_opens_at(host, day: date) -> datetime:
    """The instant the sweep becomes due, from the seam's own close.

    Derived rather than hard-coded on purpose - see
    `test_the_desk_resolves_its_local_zone_as_a_fixed_offset` below.
    """
    close = host.actual_session_close(_at(day, "12:00"))
    return (close + timedelta(minutes=host.SWEEP_AFTER_SCAN_WINDOW_MINUTES)).replace(tzinfo=None)


def test_a_half_day_closes_three_hours_earlier_in_eastern():
    """The claim is about the exchange, so it is asserted in the exchange's zone.

    Asserting it in desk-local time would be asserting the desk's own DST
    handling as well, and on this machine that is a fixed offset (see below).
    """
    from market_calendar import MARKET_TZ

    host = _host()
    half = host.actual_session_close(_at(HALF_DAY, "12:00")).astimezone(MARKET_TZ)
    regular = host.actual_session_close(_at(MONDAY, "12:00")).astimezone(MARKET_TZ)
    assert half.time() == ec.EARLY_CLOSE
    assert regular.time() == ec.REGULAR_CLOSE
    assert regular.hour - half.hour == 3


def test_a_half_day_sweeps_at_its_own_close_plus_thirty_five():
    host = _host()
    opens = _sweep_opens_at(host, HALF_DAY)
    assert host._after_close_jobs_due(opens - timedelta(minutes=1))["sweep"] is False
    assert host._after_close_jobs_due(opens)["sweep"] is True
    # ...and a REGULAR day at that same wall clock does not sweep, which is the
    # direction that would have put the sweep inside the scan window.
    regular_at_half_days_time = _at(MONDAY, opens.strftime("%H:%M"))
    assert _host()._after_close_jobs_due(regular_at_half_days_time)["sweep"] is False


def test_standard_time_after_the_dst_change_still_lands_at_close_plus_thirty_five():
    """2026-11-30 is ET standard time and a regular session."""
    host = _host()
    opens = _sweep_opens_at(host, POST_DST)
    assert host._after_close_jobs_due(opens - timedelta(minutes=1))["sweep"] is False
    assert host._after_close_jobs_due(opens)["sweep"] is True


def test_the_desk_resolves_its_local_zone_as_a_fixed_offset():
    """A pre-existing `market_session` property, recorded rather than papered over.

    On Windows `datetime.now().astimezone().tzinfo` has no IANA key, so
    `get_market_local_timezone` falls through to that **fixed-offset** object -
    today's offset, applied to every date. A session window for a November date
    therefore reads -07:00 on a desk that is currently on PDT, an hour off in
    wall-clock terms.

    It does not reach this scheduler, which only ever compares *now* against
    *today's* close, where today's offset is by definition the right one. It
    would reach anything that reasons about a session months away, and fixing it
    changes displayed labels and slot times across the desk - a blast radius
    outside this packet. Flagged in the checkpoint, not silently changed.
    """
    from market_session import get_market_local_timezone

    tz, _name = get_market_local_timezone()
    if hasattr(tz, "key"):
        pytest.skip("this desk resolves a named IANA zone; the finding does not apply")
    august = datetime(2026, 8, 24, 12).replace(tzinfo=tz).utcoffset()
    november = datetime(2026, 11, 30, 12).replace(tzinfo=tz).utcoffset()
    assert august == november, "a fixed offset does not change across the DST boundary"


def test_a_weekend_is_never_due():
    due = _host()._after_close_jobs_due(_at(date(2026, 8, 22), "18:00"))
    assert due["sweep"] is False and due["refresh"] is False
    assert due["reason"] == "weekend"


def test_an_unreadable_session_defers_both_rather_than_guessing(monkeypatch):
    from bounce_bot_lib.legacy import BounceBot

    host = _host()
    host.actual_session_close = lambda now=None: None
    due = BounceBot._after_close_jobs_due(host, _at(MONDAY, "13:40"))
    assert due["sweep"] is False and due["refresh"] is False
    assert due["reason"] == "session_unreadable"


def test_the_worker_does_not_start_twice(monkeypatch):
    """A second worker would put two sweeps on the same machine-wide lock."""
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    source = inspect.getsource(BounceBot._maybe_refresh_learning_after_close)
    assert "_after_close_worker_running" in source
    assert "finally:" in source, "the flag is cleared even when the worker raises"
