"""TJ-10 item 2 - the verdict is MEASURED, and an open horizon is never zero.

Packet `.claude/packets/TJ-10.md` item 2; `plan.md` §12.4 "TJ-10" items 1, 2 and
5; decision 0021 answer 14 ("'Were you right' is a measured row, never a model's
opinion"). RED before the build.

The contract these tests pin
----------------------------

``grade_read(row, *, m5_bars=(), daily_bars=(), atr=None, now) -> dict``
    PURE. Every bar it is allowed to see is handed in; it opens no store, has
    no clock of its own, and calls no model.

    ``{"grade_id", "read_id", "entry_id", "session", "horizon", "verdict",
       "anchor_price", "anchor_at", "final_price", "final_at", "move",
       "move_atr", "checkpoints", "supersedes", "graded_at"}``

    * ``rest_of_day``: **the anchor is the first COMPLETED M5 bar that starts
      AFTER the stamp**, and the anchor price is that bar's ``open``. Nothing
      at or before the stamp is ever read (`plan.md` §12.3: point-in-time). It
      ends at **that session's actual close**
      (`market_early_close.session_close`) - 13:00 Eastern on a scheduled half
      day, never 16:00.
    * ``next_5_sessions``: the anchor is the **decision session's own daily
      close** (the last daily close the trader could have known at the stamp,
      for an in-session read and for an after-close one alike - TJ-11F), and
      the checkpoints are the daily closes **1, 3 and 5 exchange sessions**
      after it (``D1_CHECKPOINT_SESSIONS``). The VERDICT is the five-session
      one; 1 and 3 are reported beside it in ``checkpoints``.
    * The verdicts are ``right`` / ``wrong`` / ``flat`` / ``pending <date>`` /
      ``unmeasured:<reason>``. An OPEN horizon is ``pending`` with the date it
      matures on - never ``wrong`` and never a zero move. Missing bars are
      ``unmeasured:<reason>`` - never zero, never a guess.
    * ``Chop`` (and ``Range``) is RIGHT when the move stays inside the band and
      a directional call is ``flat`` when it does. The band is the ONE constant
      ``FLAT_BAND_ATR`` with ``FLAT_BAND_REASON`` written beside it; no test
      here asserts its VALUE, because the packet declares none (see
      ``tests/tj10_support.atr_for``).

``regrade_matured(now, *, root=None) -> list[dict]``
    The deterministic nightly hook. The lead registers it at integration; this
    module only has to expose it and it must call no model.

The store, under ``DAY_REVIEW_DIR / "reads"``
    ``reads_path(session, *, root=None)``, ``append_grades(session, rows, *,
    root=None)``, ``read_grades(session, *, root=None)``. APPEND-ONLY: a
    matured grade is a NEW row naming the old in ``supersedes``, and the first
    row stays on disk exactly as it was written.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj10_support as fx  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")
EASTERN = ZoneInfo("America/New_York")


# LEAD AMENDMENT 2026-09-20: `_read()` builds a CLICKED read, and the lead's review
# decision made the store refuse a gradable CLICK whose context is only the named
# absence (a clicked read is the only evidence TJ-16 will ever have). Three tests
# below store a first grade built with no context at all; they now hand
# `grade_read` a real snapshot. Not one assertion in them moved.
_A_REAL_SNAPSHOT = {"internals": {}, "direction": "up"}


def _read(direction: str, horizon: str = "rest_of_day", timeframe: str = "M5"):
    import market_read_grades as grades

    entry = fx.mentor_entry(
        direction=direction, horizon=horizon, timeframe=timeframe
    )
    return grades.read_rows([entry], session=fx.SESSION)[0]


# -- the anchor --------------------------------------------------------------


def test_the_anchor_is_the_first_completed_bar_after_the_stamp_and_never_the_stamps_own():
    """07:02 sits inside bar 6 (99.0). The read is measured from bar 7's open."""
    import market_read_grades as grades

    grade = grades.grade_read(
        _read("up"),
        m5_bars=fx.session_tape(),
        atr=fx.atr_for(fx.SESSION_MOVE, 3.0),
        now=fx.AFTER_THE_CLOSE,
    )

    assert grade["anchor_price"] == fx.ANCHOR_LEVEL  # 100.0, not 99.0
    assert datetime.fromisoformat(grade["anchor_at"]) == (
        fx.FIRST_BAR + timedelta(minutes=5 * fx.ANCHOR_BAR)
    )
    assert grade["final_price"] == fx.CLOSE_LEVEL
    assert grade["move"] == pytest.approx(fx.SESSION_MOVE)


def test_a_forming_last_bar_is_not_the_close_and_the_read_stays_pending():
    """At 12:58 the 12:55 bar has not finished. Completed bars only."""
    import market_read_grades as grades

    grade = grades.grade_read(
        _read("up"),
        m5_bars=fx.session_tape(),
        atr=fx.atr_for(fx.SESSION_MOVE, 3.0),
        now=fx.BEFORE_THE_BELL,
    )

    assert grade["verdict"] == f"pending {fx.SESSION}"
    assert grade["final_price"] is None
    assert grade["move"] is None


# -- rest of day -------------------------------------------------------------


def test_an_up_call_that_ran_three_band_widths_up_is_right():
    import market_read_grades as grades

    grade = grades.grade_read(
        _read("up"),
        m5_bars=fx.session_tape(),
        atr=fx.atr_for(fx.SESSION_MOVE, 3.0),
        now=fx.AFTER_THE_CLOSE,
    )

    assert grade["move_atr"] == pytest.approx(3.0 * grades.FLAT_BAND_ATR)
    assert grade["verdict"] == grades.VERDICT_RIGHT


def test_a_down_call_on_the_same_tape_is_wrong():
    import market_read_grades as grades

    grade = grades.grade_read(
        _read("down"),
        m5_bars=fx.session_tape(),
        atr=fx.atr_for(fx.SESSION_MOVE, 3.0),
        now=fx.AFTER_THE_CLOSE,
    )

    assert grade["verdict"] == grades.VERDICT_WRONG


def test_a_directional_call_inside_the_band_is_flat_and_never_wrong():
    """Half a band-width up. The trader was not wrong; the market did nothing."""
    import market_read_grades as grades

    grade = grades.grade_read(
        _read("up"),
        m5_bars=fx.session_tape(),
        atr=fx.atr_for(fx.SESSION_MOVE, 0.5),
        now=fx.AFTER_THE_CLOSE,
    )

    assert grade["move_atr"] == pytest.approx(0.5 * grades.FLAT_BAND_ATR)
    assert grade["verdict"] == grades.VERDICT_FLAT


def test_a_chop_call_is_right_inside_the_band_and_wrong_outside_it():
    """`Chop` is a call like any other: it is graded, and it can be wrong."""
    import market_read_grades as grades

    inside = grades.grade_read(
        _read("chop"),
        m5_bars=fx.session_tape(),
        atr=fx.atr_for(fx.SESSION_MOVE, 0.5),
        now=fx.AFTER_THE_CLOSE,
    )
    outside = grades.grade_read(
        _read("chop"),
        m5_bars=fx.session_tape(),
        atr=fx.atr_for(fx.SESSION_MOVE, 3.0),
        now=fx.AFTER_THE_CLOSE,
    )

    assert inside["verdict"] == grades.VERDICT_RIGHT
    assert outside["verdict"] == grades.VERDICT_WRONG


def test_the_band_edge_is_inside_the_band():
    """Exactly one band-width is INSIDE it - the same inclusive boundary the
    completed-bar rule uses (`scripts/completed_bars.py`)."""
    import market_read_grades as grades

    grade = grades.grade_read(
        _read("up"),
        m5_bars=fx.session_tape(),
        atr=fx.atr_for(fx.SESSION_MOVE, 1.0),
        now=fx.AFTER_THE_CLOSE,
    )

    assert grade["move_atr"] == pytest.approx(grades.FLAT_BAND_ATR)
    assert grade["verdict"] == grades.VERDICT_FLAT


def test_the_flat_band_is_one_named_constant_with_its_reason_beside_it():
    import market_read_grades as grades

    assert isinstance(grades.FLAT_BAND_ATR, float)
    assert grades.FLAT_BAND_ATR > 0
    assert len(str(grades.FLAT_BAND_REASON).strip()) > 20, (
        "the band's reason is written beside it, not left to a reader to guess"
    )


def test_a_read_with_no_atr_is_unmeasured_and_never_flat():
    """Missing data is uncertainty, never confirmation (plan.md sec 5)."""
    import market_read_grades as grades

    grade = grades.grade_read(
        _read("up"), m5_bars=fx.session_tape(), atr=None, now=fx.AFTER_THE_CLOSE
    )

    assert grade["verdict"].startswith(f"{grades.UNMEASURED_PREFIX}:")
    assert grade["verdict"] != grades.VERDICT_FLAT


def test_a_session_with_no_tape_is_unmeasured_and_never_zero():
    import market_read_grades as grades

    grade = grades.grade_read(
        _read("up"), m5_bars=(), atr=10.0, now=fx.AFTER_THE_CLOSE
    )

    assert grade["verdict"].startswith(f"{grades.UNMEASURED_PREFIX}:")
    assert grade["move"] is None


def test_a_no_view_answer_is_never_graded_right_or_wrong():
    import market_read_grades as grades

    grade = grades.grade_read(
        _read("no_view"),
        m5_bars=fx.session_tape(),
        atr=fx.atr_for(fx.SESSION_MOVE, 3.0),
        now=fx.AFTER_THE_CLOSE,
    )

    assert grade["verdict"] not in {
        grades.VERDICT_RIGHT, grades.VERDICT_WRONG, grades.VERDICT_FLAT
    }


# -- the early close ---------------------------------------------------------

HALF_DAY = "2026-11-27"  # the day after Thanksgiving 2026; closes 13:00 Eastern


def _half_day_tape():
    """06:30 Pacific to 10:55 - three bars of it PAST the 10:00 Pacific bell.

    Yahoo has handed the desk post-bell rows on a half day before, so the tape
    is written that way on purpose: a grader that takes "the last completed
    bar" without asking the calendar reads 150.0 and calls a Down read wrong.
    """
    first = datetime(2026, 11, 27, 6, 30, tzinfo=PACIFIC)
    rows = []
    for index in range(54):  # 06:30 .. 10:55
        if index < 7:
            level = 99.0
        elif index < 41:  # through the 09:55 bar, the last one before the bell
            level = 100.0
        elif index == 41:
            level = 98.0  # the 09:55 bar: the session's true last close
        else:
            level = 150.0  # after the bell, and not this session's business
        rows.append({
            "dt": first + timedelta(minutes=5 * index),
            "open": level, "high": level + 0.5, "low": level - 0.5,
            "close": level, "volume": 1_000,
        })
    return rows


def test_rest_of_day_on_a_half_day_ends_at_the_early_close():
    import market_read_grades as grades

    entry = fx.mentor_entry(
        direction="down", horizon="rest_of_day", timeframe="M5",
        session=HALF_DAY,
        created_at=datetime(2026, 11, 27, 7, 2, tzinfo=PACIFIC),
    )
    row = grades.read_rows([entry], session=HALF_DAY)[0]

    grade = grades.grade_read(
        row,
        m5_bars=_half_day_tape(),
        atr=fx.atr_for(2.0, 3.0),
        now=datetime(2026, 11, 30, 8, 0, tzinfo=PACIFIC),
    )

    assert grade["final_price"] == 98.0, "the 10:00 Pacific bell is the close"
    assert datetime.fromisoformat(grade["final_at"]).astimezone(EASTERN) <= (
        datetime(2026, 11, 27, 13, 0, tzinfo=EASTERN)
    )
    assert grade["move"] == pytest.approx(-2.0)
    assert grade["verdict"] == grades.VERDICT_RIGHT


# -- the five-session horizon ------------------------------------------------


def test_a_d1_read_is_pending_with_the_fifth_sessions_date_until_it_closes():
    """09-21 and 09-23 are already measured; the VERDICT still waits for 09-25."""
    import market_read_grades as grades

    row = _read("up", horizon="next_5_sessions", timeframe="D1")
    daily = fx.daily_bars({
        fx.SESSION: 100.0,
        fx.NEXT_SESSIONS[0]: 104.0,
        fx.NEXT_SESSIONS[1]: 105.0,
        fx.NEXT_SESSIONS[2]: 106.0,
    })

    grade = grades.grade_read(
        row, daily_bars=daily, atr=fx.atr_for(6.0, 3.0),
        now=datetime(2026, 9, 23, 17, 0, tzinfo=PACIFIC),
    )

    assert grade["verdict"] == f"pending {fx.NEXT_SESSIONS[4]}"
    assert grade["verdict"] != grades.VERDICT_RIGHT
    measured = {str(cell["session"]): cell for cell in grade["checkpoints"]}
    assert measured[fx.NEXT_SESSIONS[0]]["move"] == pytest.approx(4.0)
    assert measured[fx.NEXT_SESSIONS[2]]["move"] == pytest.approx(6.0)


def test_a_matured_d1_read_is_graded_on_the_fifth_session_close():
    import market_read_grades as grades

    row = _read("up", horizon="next_5_sessions", timeframe="D1")
    daily = fx.daily_bars({
        fx.SESSION: 100.0,
        fx.NEXT_SESSIONS[0]: 104.0,
        fx.NEXT_SESSIONS[1]: 105.0,
        fx.NEXT_SESSIONS[2]: 106.0,
        fx.NEXT_SESSIONS[3]: 107.0,
        fx.NEXT_SESSIONS[4]: 109.0,
    })

    grade = grades.grade_read(
        row, daily_bars=daily, atr=fx.atr_for(9.0, 3.0),
        now=datetime(2026, 9, 26, 8, 0, tzinfo=PACIFIC),
    )

    assert grade["final_price"] == 109.0
    assert grade["move"] == pytest.approx(9.0)
    assert grade["move_atr"] == pytest.approx(3.0 * grades.FLAT_BAND_ATR)
    assert grade["verdict"] == grades.VERDICT_RIGHT


def test_a_d1_daily_series_carrying_a_forming_bar_for_today_is_not_a_close():
    """`chart_snapshot.load_d1_bars` does NOT strip today's forming bar.

    Measured on the branch point: the loader returns every row in the parquet
    file. So the grader has to cut the series itself - a daily bar dated on or
    after `now`'s own session has not closed.
    """
    import market_read_grades as grades

    row = _read("up", horizon="next_5_sessions", timeframe="D1")
    daily = fx.daily_bars({
        fx.SESSION: 100.0,
        fx.NEXT_SESSIONS[0]: 104.0,
        fx.NEXT_SESSIONS[1]: 105.0,
        fx.NEXT_SESSIONS[2]: 106.0,
        fx.NEXT_SESSIONS[3]: 107.0,
        # 09-25 is TODAY and still trading at 10:00 Pacific.
        fx.NEXT_SESSIONS[4]: 130.0,
    })

    grade = grades.grade_read(
        row, daily_bars=daily, atr=fx.atr_for(9.0, 3.0),
        now=datetime(2026, 9, 25, 10, 0, tzinfo=PACIFIC),
    )

    assert grade["verdict"] == f"pending {fx.NEXT_SESSIONS[4]}"
    assert grade["final_price"] != 130.0


def test_a_range_call_over_five_sessions_is_right_inside_the_band():
    import market_read_grades as grades

    row = _read("range", horizon="next_5_sessions", timeframe="D1")
    daily = fx.daily_bars({
        fx.SESSION: 100.0,
        fx.NEXT_SESSIONS[0]: 100.4,
        fx.NEXT_SESSIONS[1]: 99.6,
        fx.NEXT_SESSIONS[2]: 100.2,
        fx.NEXT_SESSIONS[3]: 99.8,
        fx.NEXT_SESSIONS[4]: 100.5,
    })

    grade = grades.grade_read(
        row, daily_bars=daily, atr=fx.atr_for(0.5, 0.5),
        now=datetime(2026, 9, 26, 8, 0, tzinfo=PACIFIC),
    )

    assert grade["verdict"] == grades.VERDICT_RIGHT


# -- the store ---------------------------------------------------------------


def test_a_matured_grade_is_a_new_row_naming_the_old_and_rewrites_nothing(tmp_path):
    import market_read_grades as grades

    row = _read("up", horizon="next_5_sessions", timeframe="D1")
    partial = fx.daily_bars({fx.SESSION: 100.0, fx.NEXT_SESSIONS[0]: 104.0})
    full = fx.daily_bars({
        fx.SESSION: 100.0,
        fx.NEXT_SESSIONS[0]: 104.0,
        fx.NEXT_SESSIONS[1]: 105.0,
        fx.NEXT_SESSIONS[2]: 106.0,
        fx.NEXT_SESSIONS[3]: 107.0,
        fx.NEXT_SESSIONS[4]: 109.0,
    })

    pending = grades.grade_read(
        row, daily_bars=partial, atr=fx.atr_for(9.0, 3.0),
        context=_A_REAL_SNAPSHOT,
        now=datetime(2026, 9, 21, 17, 0, tzinfo=PACIFIC),
    )
    grades.append_grades(fx.SESSION, [pending], root=tmp_path)

    matured = grades.grade_read(
        row, daily_bars=full, atr=fx.atr_for(9.0, 3.0),
        now=datetime(2026, 9, 26, 8, 0, tzinfo=PACIFIC),
        supersedes=pending["grade_id"],
    )
    grades.append_grades(fx.SESSION, [matured], root=tmp_path)

    stored = grades.read_grades(fx.SESSION, root=tmp_path)
    assert len(stored) == 2, "an append-only store keeps the first answer"
    assert stored[0]["verdict"] == f"pending {fx.NEXT_SESSIONS[4]}"
    assert stored[1]["verdict"] == grades.VERDICT_RIGHT
    assert stored[1]["supersedes"] == pending["grade_id"]
    assert stored[0]["supersedes"] == ""


def test_the_reads_store_lives_under_the_day_review_directory(tmp_path):
    import market_read_grades as grades

    path = grades.reads_path(fx.SESSION, root=tmp_path)

    assert path.parent == Path(tmp_path) / "reads"
    assert fx.SESSION in path.name


def test_the_nightly_hook_matures_a_pending_row_into_a_new_superseding_row(tmp_path):
    """`regrade_matured(now)` - the ONE function the lead registers (packet).

    Deterministic and modelless: it re-reads the bars, re-grades what has
    matured, and appends. Nothing already on disk moves.
    """
    import market_read_grades as grades

    row = _read("up", horizon="next_5_sessions", timeframe="D1")
    pending = grades.grade_read(
        row,
        daily_bars=fx.daily_bars({fx.SESSION: 100.0, fx.NEXT_SESSIONS[0]: 104.0}),
        context=_A_REAL_SNAPSHOT,
        atr=fx.atr_for(9.0, 3.0),
        now=datetime(2026, 9, 21, 17, 0, tzinfo=PACIFIC),
    )
    grades.append_grades(fx.SESSION, [pending], root=tmp_path)

    full = fx.daily_bars({
        fx.SESSION: 100.0,
        fx.NEXT_SESSIONS[0]: 104.0,
        fx.NEXT_SESSIONS[1]: 105.0,
        fx.NEXT_SESSIONS[2]: 106.0,
        fx.NEXT_SESSIONS[3]: 107.0,
        fx.NEXT_SESSIONS[4]: 109.0,
    })
    written = grades.regrade_matured(
        datetime(2026, 9, 26, 2, 0, tzinfo=PACIFIC),
        root=tmp_path,
        daily_bars_for=lambda _symbol: full,
        atr_for=lambda _symbol, _session: fx.atr_for(9.0, 3.0),
    )

    assert len(written) == 1, "the matured horizon was not re-graded"
    stored = grades.read_grades(fx.SESSION, root=tmp_path)
    assert len(stored) == 2
    assert stored[0]["verdict"] == f"pending {fx.NEXT_SESSIONS[4]}"
    assert stored[1]["verdict"] == grades.VERDICT_RIGHT
    assert stored[1]["supersedes"] == pending["grade_id"]


def test_the_nightly_hook_is_idempotent_and_never_re_supersedes_a_closed_row(tmp_path):
    """A closed horizon is closed. A second night writes nothing."""
    import market_read_grades as grades

    row = _read("up", horizon="rest_of_day", timeframe="M5")
    closed = grades.grade_read(
        row,
        m5_bars=fx.session_tape(),
        context=_A_REAL_SNAPSHOT,
        atr=fx.atr_for(fx.SESSION_MOVE, 3.0),
        now=fx.AFTER_THE_CLOSE,
    )
    assert closed["verdict"] == grades.VERDICT_RIGHT
    grades.append_grades(fx.SESSION, [closed], root=tmp_path)

    written = grades.regrade_matured(
        datetime(2026, 9, 26, 2, 0, tzinfo=PACIFIC),
        root=tmp_path,
        m5_bars_for=lambda _symbol, _session: fx.session_tape(),
        atr_for=lambda _symbol, _session: fx.atr_for(fx.SESSION_MOVE, 3.0),
    )

    assert written == []
    assert len(grades.read_grades(fx.SESSION, root=tmp_path)) == 1


def test_grading_never_starts_a_thread_or_sleeps(monkeypatch):
    """Pure arithmetic. The post-close tick owns the worker, not this module."""
    import threading
    import time

    import market_read_grades as grades

    def _banned(*_args, **_kwargs):
        raise AssertionError("the pure grader started a thread or slept")

    monkeypatch.setattr(threading, "Thread", _banned)
    monkeypatch.setattr(time, "sleep", _banned)

    grades.grade_read(
        _read("up"),
        m5_bars=fx.session_tape(),
        atr=fx.atr_for(fx.SESSION_MOVE, 3.0),
        now=fx.AFTER_THE_CLOSE,
    )


def test_a_friday_night_call_is_read_on_fridays_session_not_mondays():
    """TJ-11F, trader 2026-09-19: an after-close call belongs to the session it
    JUDGED. The desk stamps 21:04 Pacific Friday with New York's SATURDAY date,
    so the row's own `session_date` is a day no session ever traded."""
    import market_calendar
    import market_read_grades as grades

    friday_night = datetime(2026, 9, 18, 21, 4, tzinfo=PACIFIC)
    entry = fx.mentor_entry(
        direction="down", horizon="next_5_sessions", timeframe="D1",
        created_at=friday_night, session="2026-09-19",
    )
    assert entry["session_date"] == "2026-09-19"
    assert market_calendar.decision_session(friday_night) == date(2026, 9, 18)

    rows = grades.read_rows([entry], session=fx.SESSION)

    assert len(rows) == 1, "Friday evening's D1 call never reached Friday's page"
    assert rows[0]["session"] == fx.SESSION
