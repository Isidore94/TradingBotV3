"""TJ-11F - an after-close decision belongs to the session it JUDGED. RED first.

Packet `.claude/packets/TJ-11F.md`. The trader, 2026-09-19 ~16:20 PDT, answering
the lead's TJ-11 question:

    *"a veto on friday night (after the market close) should not be considered
    monday since we have new information then."*

This REVERSES the last clause of decision 0021 answer 16 as TJ-11 built it. A
call made after Friday's close, or over the weekend, is a judgement on FRIDAY's
information - Friday's scan, Friday's close. Monday's scan is new information
and the decision does not carry to it.

The contract these tests pin, so the builder has nothing to guess
-----------------------------------------------------------------

``scripts/market_calendar.py``
    ``decision_session(stamp) -> date | None`` is the ONE seam and it now maps
    BACKWARD. The rule, once: **the exchange session whose New York calendar
    date the stamp falls on** when that date is a session day - pre-market,
    in-session and after the close all stay on that day - **else the most
    recent PRIOR session** (a weekend, a holiday, and the small hours of a
    non-session date: Friday 21:04 Pacific is Saturday 00:04 Eastern, and it is
    Friday's judgement). Aware stamps are converted with ``astimezone`` and
    never stripped; a naive stamp is market-local, which is what the desk
    writes; an unreadable stamp answers ``None`` rather than guessing.
    ``next_session`` stays where it is - this packet does not touch it.

``scripts/ui/annotations/store.py``
    A NEW row carries ``decision_session`` (the judged session) **and**
    ``decision_session_rule: "judged_session_v2"``. The marker exists because
    rows written between wave 1 going live and this fix could hold a
    FORWARD-mapped value: a row whose ``decision_session`` is present WITHOUT
    the marker was written under the old rule, so every reader IGNORES the
    stored value and recomputes from the row's own stamp. A row WITH the marker
    is believed. Old rows are never rewritten.
    ``session_date`` stays byte-identical to base for every writer and every
    reader - `pick_feedback`, `review_learning`, the three cohort graders and
    `daily_recap_reader._decisions` all join on it by exact match, and the
    parity tests in ``tests/test_tj11_session_stamp_is_additive.py`` stay green
    unchanged.

Measured read-only on a COPY of ``C:\\TradingBotData\\trader_annotations.jsonl``
(2026-09-19): 1,198 rows, **0** carry ``decision_session`` at all, **18** carry
New York's Saturday date 2026-09-19 (12 vetoes, 6 claimed likes, all D1, stamped
2026-09-18 21:04-21:07 Pacific), and **0** fall in the Sunday-evening-Pacific /
Monday-New-York-date edge.

Plain dicts and ``tmp_path`` only: no live store, no clock, no network.
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

EASTERN = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")

#: Friday 2026-09-18, 21:04:28 Pacific - the live case, byte-for-byte the
#: `created_at` of the first of the 18 rows on the desk.
FRIDAY_EVENING = datetime(2026, 9, 18, 21, 4, 28, 734007, tzinfo=PACIFIC)
FRIDAY = date(2026, 9, 18)
SATURDAY = date(2026, 9, 19)
SUNDAY = date(2026, 9, 20)
MONDAY = date(2026, 9, 21)

#: The rule marker a row written by THIS build carries. It is a schema stamp,
#: not a vocabulary version: the packet names the string and the readers key on
#: it, so it is asserted literally on purpose.
RULE_V2 = "judged_session_v2"


def _decision_session(stamp):
    from market_calendar import decision_session

    return decision_session(stamp)


def _a_veto_code() -> str:
    """A code the shipped vocabulary really has. Never a literal version."""
    from ui.annotations.vocabulary import load_veto_vocabulary

    return list(load_veto_vocabulary().codes)[0]


# -- the rule ----------------------------------------------------------------


def test_a_friday_evening_call_after_the_close_is_fridays_judgement():
    """The live case: 18 D1 calls made 2026-09-18 21:04-21:07 Pacific.

    21:04 Pacific is 00:04 Eastern on the Saturday, which is why the desk
    stamped them 2026-09-19. The information they were made on is Friday's.
    """
    assert _decision_session(FRIDAY_EVENING) == FRIDAY
    assert _decision_session(datetime(2026, 9, 18, 21, 30, tzinfo=PACIFIC)) == FRIDAY


def test_a_call_after_the_close_on_a_session_day_stays_on_that_day():
    """16:30 Eastern on a Friday is Friday's judgement, not Monday's."""
    assert _decision_session(datetime(2026, 9, 18, 16, 30, tzinfo=EASTERN)) == FRIDAY
    # And a weekday evening keeps the weekday it was made on.
    assert _decision_session(datetime(2026, 9, 17, 17, 0, tzinfo=EASTERN)) == date(2026, 9, 17)


def test_a_weekend_stamp_is_read_as_the_session_before_it():
    """A Saturday or a Sunday belongs BACK to Friday, not forward to Monday."""
    assert _decision_session("2026-09-19") == FRIDAY
    assert _decision_session(SATURDAY) == FRIDAY
    assert _decision_session(SUNDAY) == FRIDAY
    assert _decision_session(datetime(2026, 9, 20, 12, 0, tzinfo=PACIFIC)) == FRIDAY


def test_a_call_over_the_labor_day_weekend_belongs_to_the_friday_before_it():
    """Labor Day 2026 is Monday 2026-09-07; the session before it is 09-04."""
    assert _decision_session(datetime(2026, 9, 4, 21, 30, tzinfo=PACIFIC)) == date(2026, 9, 4)
    assert _decision_session("2026-09-05") == date(2026, 9, 4)
    assert _decision_session("2026-09-06") == date(2026, 9, 4)
    assert _decision_session("2026-09-07") == date(2026, 9, 4)
    # The session after the holiday is its own.
    assert _decision_session("2026-09-08") == date(2026, 9, 8)


def test_a_call_inside_or_before_a_session_keeps_that_session():
    """Pre-market, the open and mid-session are all that day - unchanged."""
    assert _decision_session(datetime(2026, 9, 21, 6, 0, tzinfo=EASTERN)) == MONDAY
    assert _decision_session(datetime(2026, 9, 18, 10, 0, tzinfo=EASTERN)) == FRIDAY
    assert _decision_session("2026-09-18") == FRIDAY


def test_an_aware_stamp_is_converted_and_never_stripped_of_its_zone():
    """22:00 Eastern on the Sunday is 02:00 UTC on the Monday.

    Strip the zone and the date reads 2026-09-21, a session, and the answer
    would be Monday's. Converted with ``astimezone`` it is Sunday evening in
    New York, so it is Friday's judgement.
    """
    sunday_evening_ny = datetime(2026, 9, 21, 2, 0, tzinfo=timezone.utc)

    assert sunday_evening_ny.astimezone(EASTERN).date() == SUNDAY
    assert _decision_session(sunday_evening_ny) == FRIDAY
    # And the live stamp again, as UTC text this time.
    assert _decision_session("2026-09-19T04:04:28.734007+00:00") == FRIDAY


def test_a_naive_stamp_is_read_as_market_local_and_never_as_utc():
    """Naive 2026-09-18 21:30 is a Friday evening in New York."""
    assert _decision_session(datetime(2026, 9, 18, 21, 30)) == FRIDAY
    assert _decision_session(datetime(2026, 9, 18, 10, 0)) == FRIDAY


def test_a_sunday_evening_pacific_stamp_is_still_mondays_new_york_date():
    """The edge the packet names, REPORTED not decided: 0 live rows are here.

    23:00 Pacific on the Sunday is 02:00 Eastern on the Monday - a session
    date, before its open - so the rule above answers Monday, exactly as the
    base `session_date` does. Pinned so the backward mapping is not taken too
    far: only a NON-session New York date walks back.
    """
    assert _decision_session(datetime(2026, 9, 20, 23, 0, tzinfo=PACIFIC)) == MONDAY


@pytest.mark.parametrize("bad", ["", None, "not-a-date"])
def test_an_unreadable_stamp_answers_none_rather_than_guessing(bad):
    assert _decision_session(bad) is None


def test_next_session_is_untouched_by_this_packet():
    """The forward helper stays; this packet only stops `decision_session`
    from being its caller."""
    from market_calendar import next_session

    assert next_session(FRIDAY) == MONDAY
    assert next_session(date(2026, 9, 4)) == date(2026, 9, 8)


# -- the writer --------------------------------------------------------------


def test_a_new_row_records_the_judged_session_and_says_which_rule_wrote_it(monkeypatch):
    """One key changes value, one key is added, `session_date` does not move."""
    import market_session
    from ui.annotations import store

    # Pinned to the Saturday the live rows carry, so the answer cannot come
    # from the suite happening to run on the right day.
    monkeypatch.setattr(
        market_session, "get_market_session_window",
        lambda *a, **k: type("_W", (), {"market_date": SATURDAY})(),
    )

    row = store.build_annotation(
        store.EVENT_VETO,
        symbol="HLIT",
        side="SHORT",
        reason_code=_a_veto_code(),
        timeframe="D1",
        created_at=FRIDAY_EVENING,
    )

    assert row[store.DECISION_SESSION_FIELD] == "2026-09-18"
    assert row["decision_session_rule"] == RULE_V2
    # EXACTLY what base writes, and it stays that way: every live reader on the
    # desk joins on this by exact match.
    assert row["session_date"] == "2026-09-19"
    assert set(row) - {store.DECISION_SESSION_FIELD, "decision_session_rule"} == {
        "schema_version", "event_id", "event_type", "symbol", "session_date",
        "created_at", "source", "reason_code", "vocab_version", "side",
        "timeframe",
    }


def test_a_same_day_after_close_call_reads_the_same_from_the_row_and_the_reader():
    """TJ-11's review found the stored field and the page's tag disagreeing.

    A veto at 16:30 Eastern on a Friday: the row said Monday, the page said
    Friday. Both say Friday now.
    """
    from ui.annotations import store

    row = store.build_annotation(
        store.EVENT_VETO,
        symbol="HLIT",
        side="SHORT",
        reason_code=_a_veto_code(),
        timeframe="D1",
        session_date="2026-09-18",
        created_at=datetime(2026, 9, 18, 16, 30, tzinfo=EASTERN),
    )

    assert row["session_date"] == "2026-09-18"
    assert row[store.DECISION_SESSION_FIELD] == "2026-09-18"
    assert store.row_decision_session(row) == "2026-09-18"


# -- the reader's two kinds of row -------------------------------------------


def _forward_rule_row() -> dict:
    """A row as wave 1 wrote it: a FORWARD `decision_session`, no marker."""
    return {
        "schema_version": 1, "event_id": "veto-1", "event_type": "veto",
        "symbol": "HLIT", "side": "SHORT", "session_date": "2026-09-19",
        "timeframe": "D1", "created_at": FRIDAY_EVENING.isoformat(),
        "source": "chart_review", "reason_code": "extended", "vocab_version": 3,
        "decision_session": "2026-09-21",
    }


def test_a_row_written_under_the_forward_rule_is_recomputed_not_believed():
    """0 live rows carry one today, but the desk could have written some.

    The value is present and it is Monday's; there is no `decision_session_rule`
    marker, so it was written by the rule this packet reverses and the reader
    recomputes it from the row's own stamp. The row itself is never rewritten.
    """
    from ui.annotations import store

    row = _forward_rule_row()

    assert store.row_decision_session(row) == "2026-09-18"
    # Untouched: a reader maps, it does not repair.
    assert row["decision_session"] == "2026-09-21"


def test_a_row_stamped_with_the_v2_marker_is_believed():
    """A marked row states its own answer and the reader takes it.

    The stored value here is deliberately one the recompute would NOT produce,
    so "believed" is proven rather than coincidental.
    """
    from ui.annotations import store

    row = _forward_rule_row() | {"decision_session_rule": RULE_V2}

    assert store.row_decision_session(row) == "2026-09-21"


def test_a_row_written_before_wave_1_is_mapped_back_from_its_stamp():
    """All 1,198 rows on the desk today are this shape: no new key at all."""
    from ui.annotations import store

    row = _forward_rule_row()
    row.pop("decision_session")

    assert store.row_decision_session(row) == "2026-09-18"
