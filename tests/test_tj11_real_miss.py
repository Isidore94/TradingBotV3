"""TJ-11 Part A - the real-miss rule and the session stamp. RED before the build.

Packet `.claude/packets/TJ-11.md` items 1 and 7; `plan.md` §12.4 "TJ-11" items 3
and 5; decision 0021 answer 16.

Nothing here reads a store, a clock or the network: every bar is a plain dict and
every moment is explicit and zoned.

The contract these tests pin, so the builder has nothing to guess
-----------------------------------------------------------------

``scripts/real_miss.py`` - one small, import-light module:

``REAL_MISS_V1``
    The rule's version name, a module constant. Carried by anything that stores
    a verdict, so a later rule can never be mistaken for this one.
``RUN_ATR = 1.0`` and ``ADVERSE_ATR = 0.5``
    The packet's two numbers, named once.
``verdict(bars, *, stamp, side, atr, now=None, bar_minutes=None) -> str``
    Returns ``"run"``, ``"no_run"``, or ``"unmeasured:<reason>"`` - never a
    number, never a bool, never ``""``.

    * ``bars`` is oldest-first, each a Mapping with ``dt`` (ISO text or a
      ``datetime``) and ``open`` / ``high`` / ``low`` / ``close``.
    * Only bars that START strictly after ``stamp`` count; the reference price
      is the OPEN of the first such bar (the rule `walkaway_day._after_move`
      already uses).
    * With ``now`` and ``bar_minutes`` given, a bar that has not completed at
      ``now`` is excluded (`scripts/completed_bars.py`'s one rule).
    * ``side`` is ``"LONG"`` or ``"SHORT"``; favourable is the high for a long
      and the low for a short.
    * A run is favourable excursion >= ``RUN_ATR`` x ``atr`` reached BEFORE
      adverse excursion >= ``ADVERSE_ATR`` x ``atr``.
    * Inside ONE bar the order of the two extremes is unknown, so the adverse
      one is taken first: missing data is uncertainty, never confirmation.
    * ``atr`` missing, non-positive or unreadable is ``unmeasured``, never a
      verdict and never zero.

``scripts/market_calendar.py`` gains the pure session stamp (packet: "put it
beside the calendar helpers, not in the annotation store"):

``decision_session(stamp) -> datetime.date``
    The exchange session a decision belongs to. A ``datetime`` (aware or naive,
    naive read as market-local), a ``date`` or ``"YYYY-MM-DD"`` text all work.
    The session the stamp falls in - that calendar date when it is a session and
    the stamp is at or before that session's close, so pre-market counts - or
    else the NEXT exchange session. Compared with ``astimezone``; a zone is
    never stripped.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PACIFIC = ZoneInfo("America/Los_Angeles")
EASTERN = ZoneInfo("America/New_York")

SESSION = "2026-09-18"
#: Every fixture below opens at 100.00 with a daily ATR(14) of 2.00, so one ATR
#: is exactly 2% and every threshold in this file is a round number.
ATR = 2.0
STAMP = datetime(2026, 9, 18, 9, 35, tzinfo=EASTERN)


def _bar(hour: int, minute: int, *, open_: float, high: float, low: float, close: float) -> dict:
    return {
        "dt": datetime(2026, 9, 18, hour, minute, tzinfo=EASTERN).isoformat(),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }


def _verdict(bars, *, side: str = "LONG", atr: float | None = ATR, **kwargs) -> str:
    import real_miss

    return real_miss.verdict(bars, stamp=STAMP, side=side, atr=atr, **kwargs)


# -- the rule ---------------------------------------------------------------


def test_the_rule_is_versioned_and_carries_the_packets_two_thresholds():
    import real_miss

    assert isinstance(real_miss.REAL_MISS_V1, str) and real_miss.REAL_MISS_V1
    assert real_miss.RUN_ATR == 1.0
    assert real_miss.ADVERSE_ATR == 0.5


def test_a_pop_that_fades_before_one_atr_is_not_a_real_run():
    """+0.75 ATR, then away. A glance calls it a miss; the rule does not."""
    bars = [
        _bar(9, 40, open_=100.0, high=101.5, low=100.0, close=101.0),
        _bar(9, 45, open_=101.0, high=101.0, low=99.5, close=99.6),
        _bar(9, 50, open_=99.6, high=99.8, low=98.5, close=98.6),
    ]

    assert _verdict(bars) == "no_run"


def test_a_name_that_goes_against_you_first_is_not_a_real_run():
    """-0.7 ATR in the first bar, +2.5 ATR in the second. Still not a miss."""
    bars = [
        _bar(9, 40, open_=100.0, high=100.2, low=98.6, close=99.0),
        _bar(9, 45, open_=99.0, high=105.0, low=99.0, close=104.5),
    ]

    assert _verdict(bars) == "no_run"


def test_a_clean_run_to_one_atr_before_half_an_atr_against_is_a_real_run():
    bars = [
        _bar(9, 40, open_=100.0, high=101.2, low=99.5, close=101.0),
        _bar(9, 45, open_=101.0, high=103.0, low=100.8, close=102.8),
    ]

    assert _verdict(bars) == "run"


def test_a_short_decision_is_measured_downward_and_the_same_tape_fails_a_long():
    bars = [
        _bar(9, 40, open_=100.0, high=100.4, low=99.9, close=100.0),
        _bar(9, 45, open_=100.0, high=100.0, low=97.5, close=97.8),
    ]

    assert _verdict(bars, side="SHORT") == "run"
    assert _verdict(bars, side="LONG") == "no_run"


def test_one_bar_that_touches_both_thresholds_is_not_a_real_run():
    """Inside a bar the order is unknown, so the adverse extreme is taken first."""
    bars = [_bar(9, 40, open_=100.0, high=102.5, low=98.5, close=101.0)]

    assert _verdict(bars) == "no_run"


def test_only_completed_bars_count_toward_a_real_run():
    """The bar that reaches +1.5 ATR is still forming at 09:48."""
    bars = [
        _bar(9, 40, open_=100.0, high=101.0, low=99.8, close=100.9),
        _bar(9, 45, open_=100.9, high=103.0, low=100.9, close=102.9),
    ]
    now = datetime(2026, 9, 18, 9, 48, tzinfo=EASTERN)

    assert _verdict(bars, now=now, bar_minutes=5) == "no_run"
    assert _verdict(bars) == "run"


def test_a_missing_atr_is_unmeasured_and_never_a_verdict():
    bars = [_bar(9, 40, open_=100.0, high=110.0, low=99.0, close=109.0)]

    for missing in (None, 0.0, "", float("nan")):
        answer = _verdict(bars, atr=missing)
        assert answer.startswith("unmeasured:"), (missing, answer)
        assert "atr" in answer.split(":", 1)[1].lower(), (missing, answer)


def test_no_bar_after_the_stamp_is_unmeasured_and_never_no_run():
    """Nothing to look at is uncertainty, not a verdict of "did not run"."""
    bars = [_bar(9, 30, open_=100.0, high=110.0, low=99.0, close=109.0)]

    answer = _verdict(bars)
    assert answer.startswith("unmeasured:"), answer
    assert answer != "no_run"


def test_the_rule_stays_import_light_for_the_nightly_slot():
    """TJ-15's nightly slot imports this; it must not drag pandas or Qt in."""
    import subprocess

    probe = (
        "import sys; sys.path.insert(0, %r); import real_miss;"
        "heavy=[n for n in ('pandas','PySide6','pyqtgraph','yfinance','ibapi')"
        " if n in sys.modules];"
        "print(','.join(heavy))" % str(SCRIPTS_DIR)
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", out.stdout


# -- the session stamp ------------------------------------------------------


def _decision_session(stamp):
    from market_calendar import decision_session

    return decision_session(stamp)


def test_a_friday_evening_call_after_the_close_is_fridays_judgement():
    """The live case: 18 D1 calls made 2026-09-18 21:04 Pacific.

    UPDATED by TJ-11F on the trader's word (2026-09-19): *"a veto on friday
    night (after the market close) should not be considered monday since we
    have new information then."* These pinned Monday under TJ-11.
    """
    assert _decision_session(datetime(2026, 9, 18, 21, 30, tzinfo=PACIFIC)) == date(2026, 9, 18)
    assert _decision_session(datetime(2026, 9, 18, 21, 4, 28, tzinfo=PACIFIC)) == date(2026, 9, 18)


def test_a_saturday_stamped_row_is_read_as_fridays_session():
    """An existing row is never rewritten; its reader maps it back (TJ-11F)."""
    assert _decision_session("2026-09-19") == date(2026, 9, 18)
    assert _decision_session(date(2026, 9, 19)) == date(2026, 9, 18)
    assert _decision_session(date(2026, 9, 20)) == date(2026, 9, 18)


def test_a_call_the_evening_before_a_monday_holiday_belongs_to_the_friday_before():
    """Labor Day 2026 is Monday 2026-09-07; the session it judged is 09-04."""
    assert _decision_session(datetime(2026, 9, 4, 21, 30, tzinfo=PACIFIC)) == date(2026, 9, 4)
    assert _decision_session("2026-09-07") == date(2026, 9, 4)


def test_a_call_inside_the_session_keeps_its_own_session():
    assert _decision_session(datetime(2026, 9, 18, 10, 0, tzinfo=EASTERN)) == date(2026, 9, 18)
    # Pre-market is not "outside" - it is before that session's close.
    assert _decision_session(datetime(2026, 9, 18, 6, 0, tzinfo=EASTERN)) == date(2026, 9, 18)
    assert _decision_session("2026-09-18") == date(2026, 9, 18)


def test_a_stamp_after_a_weekday_close_keeps_its_own_session():
    """UPDATED by TJ-11F: the close is no longer a boundary."""
    assert _decision_session(datetime(2026, 9, 17, 17, 0, tzinfo=EASTERN)) == date(2026, 9, 17)


def test_a_zone_is_converted_with_astimezone_and_never_stripped():
    """UPDATED by TJ-11F: the close stopped being a boundary, so this test
    lost its old subject (16:00 vs 16:01 ET are now the same session) and
    proves the zone rule on the boundary that remains.

    22:00 Eastern on Sunday 2026-09-20 is 02:00 UTC on the Monday. Strip the
    zone and the date reads Monday, a session, and the answer would be
    Monday's. Converted it is a Sunday evening in New York, so it is Friday's.
    """
    at_the_close = datetime(2026, 9, 18, 20, 0, tzinfo=timezone.utc)
    sunday_evening_ny = datetime(2026, 9, 21, 2, 0, tzinfo=timezone.utc)

    assert _decision_session(at_the_close) == date(2026, 9, 18)
    assert sunday_evening_ny.astimezone(EASTERN).date() == date(2026, 9, 20)
    assert _decision_session(sunday_evening_ny) == date(2026, 9, 18)


def test_a_naive_stamp_is_read_as_market_local_and_never_as_utc():
    """UPDATED by TJ-11F: naive Friday 21:30 now answers Friday either way, so
    the case that still separates the two readings is the small hours.

    Naive 2026-09-21 02:00 read as market-local is Monday pre-market, a
    session. Read as UTC it would be Sunday 22:00 in New York, and the answer
    would be Friday's.
    """
    assert _decision_session(datetime(2026, 9, 21, 2, 0)) == date(2026, 9, 21)
    assert _decision_session(datetime(2026, 9, 18, 21, 30)) == date(2026, 9, 18)
    assert _decision_session(datetime(2026, 9, 18, 10, 0)) == date(2026, 9, 18)


@pytest.mark.parametrize("bad", ["", None, "not-a-date"])
def test_an_unreadable_stamp_answers_none_rather_than_guessing_today(bad):
    assert _decision_session(bad) is None
