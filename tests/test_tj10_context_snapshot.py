"""TJ-10 item 3 / TJ-16 item 1 - no graded click is stored without its context.

Packet `.claude/packets/TJ-10.md` item 3; `plan.md` §12.4 "TJ-16" item 1 ("the
context snapshot ships WITH TJ-10 so no click is ever stored without one") and
"TJ-14" item 6. RED before the build.

The contract these tests pin
----------------------------

``context_for(entry, *, row, bars=None, spy_m5_bars=(), prior_daily_bar=None,
d1_labels=None, prior_grades=(), latest_d1_click=None) -> dict``

    A POINT-IN-TIME snapshot taken at the read's stamp, from completed bars
    only, with nothing in it that happened after the stamp.

    * **The market half is `trade_mentor_context_v2`, and there is never a
      second builder.** When the entry already carries a v2 ``mentor.context``
      that block IS the snapshot's ``internals``, untouched. When it does not -
      a skipped hour, a note typed on the desk tab, or one of the 22 live rows
      carrying a v1 context - it is rebuilt through
      ``trade_mentor_context.internals_at(session, stamp, bars)`` and through
      nothing else.
    * Around it: ``hour``, ``spy_vs_session_vwap``, ``spy_vs_prior_range``,
      ``gap_pct``, ``d1_environment`` (the label of the PRIOR session, the last
      one the desk had labelled at the stamp), ``last_hour_spy``,
      ``agrees_with_own_d1``, ``previous_call_verdict``, ``confidence`` and
      ``direction``.
    * ``previous_call_verdict`` is the verdict **as known at the stamp** -
      ``right`` / ``wrong`` / ``none``. A call that was still open at the stamp
      and matured later is NOT it; that is the whole after-a-miss question.
    * A field the desk cannot measure is ``UNMEASURED`` - never 0, never None,
      never a guess.

``append_grades`` REFUSES a gradable grade with no context, because a graded
click stored without its context can never be given one afterwards.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
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
import tj14a_support as internals_fx  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")

#: 08:02 Pacific. The last COMPLETED bar starts 07:55, so bars 0..17 are in
#: scope and bar 18 onwards is the future.
LATE_STAMP = datetime(2026, 9, 18, 8, 2, tzinfo=PACIFIC)
PRIOR_SESSION = "2026-09-17"


def _split_tape():
    """Flat, then up into the stamp, then a crash AFTER it.

    Point-in-time (bars 0..17): the last hour (07:00-08:00) runs 100 -> 110, and
    110 is above any weighting of a VWAP built from those eighteen bars.
    Over the WHOLE tape the same two questions answer the other way, so a
    snapshot that read past its own stamp gets both of them wrong.
    """
    levels = [100.0] * 12 + [110.0] * 6 + [50.0] * 60
    return [
        {
            "dt": fx.FIRST_BAR + timedelta(minutes=5 * index),
            "open": level, "high": level + 0.5, "low": level - 0.5,
            "close": level, "volume": 1_000,
        }
        for index, level in enumerate(levels)
    ]


def _click(direction: str = "up", *, confidence: str = "high", context=None,
           timeframe: str = "M5", horizon: str = "rest_of_day"):
    import market_read_grades as grades

    entry = fx.mentor_entry(
        direction=direction, horizon=horizon, timeframe=timeframe,
        confidence=confidence, created_at=LATE_STAMP, context=context,
    )
    return entry, grades.read_rows([entry], session=fx.SESSION)[0]


# -- one builder -------------------------------------------------------------


def test_a_click_that_already_carries_a_v2_context_reuses_it_and_builds_nothing(
    monkeypatch,
):
    import market_read_grades as grades
    import trade_mentor_context

    stored = internals_fx.context()
    assert stored["schema"] == trade_mentor_context.SCHEMA

    def _never(*_args, **_kwargs):
        raise AssertionError("a second internals builder ran for a stored v2 block")

    monkeypatch.setattr(trade_mentor_context, "build_context", _never)
    monkeypatch.setattr(trade_mentor_context, "internals_at", _never)

    entry, row = _click(context=stored)
    snapshot = grades.context_for(entry, row=row, spy_m5_bars=_split_tape())

    assert snapshot["internals"] == stored


def test_a_row_with_a_v1_context_is_rebuilt_through_the_one_function(monkeypatch):
    """22 of the live rows carry v1. v1 is readable, and it is not v2."""
    import market_read_grades as grades
    import trade_mentor_context

    calls: list[tuple] = []
    real = trade_mentor_context.internals_at

    def _record(session, stamp, bars):
        calls.append((session, stamp))
        return real(session, stamp, bars)

    monkeypatch.setattr(trade_mentor_context, "internals_at", _record)

    entry, row = _click(
        context={"schema": trade_mentor_context.SCHEMA_V1, "readings": []}
    )
    snapshot = grades.context_for(
        entry, row=row, bars=internals_fx.bars(), spy_m5_bars=_split_tape()
    )

    assert len(calls) == 1, "the rebuild did not go through `internals_at`"
    assert snapshot["internals"]["schema"] == trade_mentor_context.SCHEMA


def test_a_row_with_no_context_at_all_is_rebuilt_rather_than_left_blank(monkeypatch):
    import market_read_grades as grades
    import trade_mentor_context

    calls: list[tuple] = []
    real = trade_mentor_context.internals_at

    def _record(session, stamp, bars):
        calls.append((session, stamp))
        return real(session, stamp, bars)

    monkeypatch.setattr(trade_mentor_context, "internals_at", _record)

    entry, row = _click(context=None)
    snapshot = grades.context_for(
        entry, row=row, bars=internals_fx.bars(), spy_m5_bars=_split_tape()
    )

    assert len(calls) == 1
    assert snapshot["internals"]["schema"] == trade_mentor_context.SCHEMA


# -- nothing after the stamp -------------------------------------------------


def test_the_snapshot_reads_no_bar_that_starts_after_the_stamp():
    """The crash at 08:00 had not happened when the trader clicked at 08:02."""
    import market_read_grades as grades

    entry, row = _click()
    snapshot = grades.context_for(
        entry, row=row, bars=internals_fx.bars(), spy_m5_bars=_split_tape()
    )

    assert snapshot["spy_vs_session_vwap"] == "above"
    assert snapshot["last_hour_spy"] == "up"


def test_the_hour_of_day_is_the_market_local_hour_of_the_stamp():
    import market_read_grades as grades

    entry, row = _click()
    snapshot = grades.context_for(
        entry, row=row, bars=internals_fx.bars(), spy_m5_bars=_split_tape()
    )

    # 08:02 Pacific is 11:02 New York, and the exchange's hour is the one every
    # other clock on this desk is stated in.
    assert snapshot["hour"] == 11


def test_the_gap_and_the_prior_range_come_from_the_prior_sessions_daily_bar():
    import market_read_grades as grades

    entry, row = _click()
    snapshot = grades.context_for(
        entry, row=row, bars=internals_fx.bars(), spy_m5_bars=_split_tape(),
        prior_daily_bar={"dt": PRIOR_SESSION, "open": 95.0, "high": 104.0,
                         "low": 94.0, "close": 98.0},
    )

    # The session opened at 100.0 (bar 0) against a 98.0 prior close: +2.0408%.
    assert snapshot["gap_pct"] == pytest.approx((100.0 - 98.0) / 98.0 * 100.0)
    # 110.0 is above the prior session's 104.0 high.
    assert snapshot["spy_vs_prior_range"] == "above"


def test_an_unmeasurable_field_says_unmeasured_and_is_never_zero():
    import market_read_grades as grades

    entry, row = _click()
    snapshot = grades.context_for(
        entry, row=row, bars=internals_fx.bars(), spy_m5_bars=_split_tape(),
        prior_daily_bar=None,
    )

    assert snapshot["gap_pct"] == grades.UNMEASURED
    assert snapshot["gap_pct"] != 0
    assert snapshot["spy_vs_prior_range"] == grades.UNMEASURED


def test_the_d1_environment_is_the_prior_sessions_label_not_this_sessions():
    """The desk labels a session from its own closed bars, so at 08:02 on the
    18th the newest label the trader could have seen is the 17th's."""
    import market_read_grades as grades

    entry, row = _click()
    snapshot = grades.context_for(
        entry, row=row, bars=internals_fx.bars(), spy_m5_bars=_split_tape(),
        d1_labels={PRIOR_SESSION: "trending_down", fx.SESSION: "trending_up"},
    )

    assert snapshot["d1_environment"] == "trending_down"


def test_a_session_nobody_labelled_is_unmeasured():
    """Measured on a copy 2026-09-19: `d1_environment.jsonl` holds 15 rows over
    five sessions and has no row at all for 2026-09-18."""
    import market_read_grades as grades

    entry, row = _click()
    snapshot = grades.context_for(
        entry, row=row, bars=internals_fx.bars(), spy_m5_bars=_split_tape(),
        d1_labels={},
    )

    assert snapshot["d1_environment"] == grades.UNMEASURED


# -- the previous call -------------------------------------------------------


def test_the_previous_calls_verdict_is_the_one_that_was_known_at_the_stamp():
    """The 06:35 call matured to `right` after the close. At 08:02 it was open,
    so the previous call the trader knew the answer to is YESTERDAY's `wrong`."""
    import market_read_grades as grades

    entry, row = _click()
    earlier_today = {
        "read_id": "r-today-0635", "session": fx.SESSION,
        "stamp": datetime(2026, 9, 18, 6, 35, tzinfo=PACIFIC).isoformat(),
        "verdict": grades.VERDICT_RIGHT,
        "graded_at": fx.AFTER_THE_CLOSE.isoformat(),
    }
    yesterday = {
        "read_id": "r-prior-1200", "session": PRIOR_SESSION,
        "stamp": datetime(2026, 9, 17, 12, 0, tzinfo=PACIFIC).isoformat(),
        "verdict": grades.VERDICT_WRONG,
        "graded_at": datetime(2026, 9, 17, 17, 0, tzinfo=PACIFIC).isoformat(),
    }

    snapshot = grades.context_for(
        entry, row=row, bars=internals_fx.bars(), spy_m5_bars=_split_tape(),
        prior_grades=[yesterday, earlier_today],
    )

    assert snapshot["previous_call_verdict"] == grades.VERDICT_WRONG


def test_a_first_ever_call_says_none_rather_than_inventing_a_previous_one():
    import market_read_grades as grades

    entry, row = _click()
    snapshot = grades.context_for(
        entry, row=row, bars=internals_fx.bars(), spy_m5_bars=_split_tape(),
        prior_grades=[],
    )

    assert snapshot["previous_call_verdict"] == "none"


def test_the_snapshot_records_the_confidence_and_the_direction_of_the_click():
    import market_read_grades as grades

    entry, row = _click("down", confidence="low")
    snapshot = grades.context_for(
        entry, row=row, bars=internals_fx.bars(), spy_m5_bars=_split_tape()
    )

    assert snapshot["confidence"] == "low"
    assert snapshot["direction"] == "down"


def test_agreement_with_the_traders_own_latest_d1_click_is_recorded():
    import market_read_grades as grades

    entry, row = _click("down")
    d1_entry, d1_row = _click(
        "down", timeframe="D1", horizon="next_5_sessions"
    )

    agrees = grades.context_for(
        entry, row=row, bars=internals_fx.bars(), spy_m5_bars=_split_tape(),
        latest_d1_click=d1_row,
    )
    up_entry, up_row = _click("up")
    disagrees = grades.context_for(
        up_entry, row=up_row, bars=internals_fx.bars(), spy_m5_bars=_split_tape(),
        latest_d1_click=d1_row,
    )
    unknown = grades.context_for(
        entry, row=row, bars=internals_fx.bars(), spy_m5_bars=_split_tape(),
        latest_d1_click=None,
    )

    assert agrees["agrees_with_own_d1"] is True
    assert disagrees["agrees_with_own_d1"] is False
    assert unknown["agrees_with_own_d1"] == grades.UNMEASURED


# -- the guard ---------------------------------------------------------------


def test_a_gradable_grade_can_never_be_stored_without_its_context(tmp_path):
    """TJ-16 item 1: the snapshot ships WITH this packet so no graded click is
    ever stored without one. A row written blank can never be given one later."""
    import market_read_grades as grades

    entry, row = _click()
    grade = grades.grade_read(
        row, m5_bars=fx.session_tape(),
        atr=fx.atr_for(fx.SESSION_MOVE, 3.0), now=fx.AFTER_THE_CLOSE,
    )
    grade["context"] = {}

    with pytest.raises(grades.ContextMissingError):
        grades.append_grades(fx.SESSION, [grade], root=tmp_path)

    assert grades.read_grades(fx.SESSION, root=tmp_path) == []
