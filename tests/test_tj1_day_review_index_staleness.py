"""TJ-1 item 4 - the staleness rule, at the two edges the bench found.

The tester pinned the middle of `day_review_index.is_stale`: a pending index
whose horizon could have matured is stale, the same index before that session
closes is not, and an index with nothing pending never is
(`tests/test_tj1_day_review_index.py`).

Running `scripts/ui/desk_bench.py` against a STAGED copy of the live home folder
found the two edges that decide whether the index is worth having at all, and the
first cut of the rule got one of them wrong:

* **A pending row whose target session had ALREADY closed when the index was
  built** is unmeasured for some other reason, and waiting will not change it.
  The first rule compared the EARLIEST pending target against today, so one such
  row made every index stale the moment it was written - the bench measured the
  page re-streaming the outcome stores on all three repeats (10.5 s settle each)
  instead of reading its own index.
* **An index built before its session closed** describes a file that is still
  being appended to, so it is always stale. Otherwise today's page would show a
  snapshot of the morning for the rest of the day.

Both are asserted on the SHAPE `build_index` really writes, not on a hand-made
dict alone: the first test here reads the fields off a built index.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SESSION = "2026-09-10"
#: Friday morning: 2026-09-10 is the last COMPLETED session.
NOW = datetime(2026, 9, 11, 7, 30)
#: The following week, long after everything in the fixture closed.
MUCH_LATER = datetime(2026, 9, 18, 8, 0)


def test_the_built_index_carries_the_fields_this_rule_reads():
    """Guard: `is_stale` reads `built_at`, `session_date`, `pending` and
    `pending_target_sessions`, so a builder that stopped writing one of them
    would make every test below vacuous."""
    import day_review_index

    index = day_review_index.build_index(SESSION, lookback_sessions=3, now=NOW)
    assert set(index) >= {
        "schema", "session_date", "built_at", "pending", "pending_target_sessions"
    }
    assert index["session_date"] == SESSION
    assert index["built_at"].startswith("2026-09-11")


def _index(**overrides):
    """An index of the fixture session, built at NOW, with nothing but the
    fields the staleness rule reads."""
    base = {
        "schema": "day_review_index_v1",
        "session_date": SESSION,
        "lookback_sessions": 3,
        "built_at": NOW.isoformat(timespec="seconds"),
        "pending": True,
        "pending_target_sessions": [],
    }
    base.update(overrides)
    return base


def test_a_pending_row_whose_session_already_closed_does_not_expire_the_index():
    """The bench defect. 2026-09-09 had closed before this index was built, so a
    pending row naming it is unmeasured for some other reason - and rebuilding
    the index every time it is opened costs the 476 MB read the index exists to
    avoid."""
    import day_review_index

    index = _index(pending_target_sessions=["2026-09-09"])
    assert day_review_index.is_stale(index, now=MUCH_LATER) is False


def test_one_row_that_could_still_mature_is_enough_to_expire_it():
    """Beside the row above, in the same index: the rule is ANY pending target
    that closed since the build, not the earliest one and not all of them."""
    import day_review_index

    index = _index(pending_target_sessions=["2026-09-09", "2026-09-11"])
    assert day_review_index.is_stale(index, now=MUCH_LATER) is True


def test_an_index_built_before_its_own_session_closed_is_always_stale():
    """Today's file is still being appended to. A snapshot of the morning shown
    at 15:00 would be a page that quietly stopped reading."""
    import day_review_index

    today = _index(
        session_date="2026-09-11",
        built_at=datetime(2026, 9, 11, 9, 0).isoformat(timespec="seconds"),
        pending_target_sessions=["2026-09-15"],
    )
    assert day_review_index.is_stale(today, now=datetime(2026, 9, 11, 9, 5)) is True
    assert day_review_index.is_stale(today, now=datetime(2026, 9, 11, 12, 0)) is True


def test_the_same_session_indexed_after_its_close_is_not(monkeypatch):
    """...and the post-close tick is what writes the one that lasts."""
    import day_review_index

    closed = _index(
        session_date="2026-09-11",
        built_at=datetime(2026, 9, 11, 17, 30).isoformat(timespec="seconds"),
        pending_target_sessions=["2026-09-15"],
    )
    assert day_review_index.is_stale(closed, now=datetime(2026, 9, 11, 18, 0)) is False


def test_a_built_at_nobody_can_read_rebuilds_rather_than_guessing():
    import day_review_index

    assert day_review_index.is_stale(_index(built_at="")) is True
    assert day_review_index.is_stale(_index(built_at="the day before")) is True


def test_a_pending_index_that_names_no_session_expires_on_the_next_close():
    """With nothing named there is nothing narrower to ask, so the honest answer
    is the coarse one - and it is still bounded by a session closing."""
    import day_review_index

    index = _index(pending_target_sessions=[])
    assert day_review_index.is_stale(index, now=NOW) is False
    assert day_review_index.is_stale(index, now=MUCH_LATER) is True


def test_a_target_that_is_not_a_date_rebuilds_rather_than_guessing():
    import day_review_index

    index = _index(pending_target_sessions=["soon"])
    assert day_review_index.is_stale(index, now=MUCH_LATER) is True


def test_the_fixture_dates_are_real_sessions():
    """Guard: the arithmetic above is the exchange calendar's, not an assumption."""
    import market_calendar

    assert market_calendar.is_session(date.fromisoformat("2026-09-09"))
    assert market_calendar.is_session(date.fromisoformat(SESSION))
    assert market_calendar.last_completed_session(NOW).isoformat() == SESSION
    assert market_calendar.last_completed_session(MUCH_LATER).isoformat() >= "2026-09-17"


@pytest.mark.parametrize("pending", [False, 0, None, ""])
def test_nothing_pending_is_never_stale_however_it_is_written(pending):
    import day_review_index

    assert day_review_index.is_stale(_index(pending=pending), now=MUCH_LATER) is False
