"""TJ-11F - the readers map BACKWARD, and the D1 ruler starts from the judged
session's close. RED before the build.

Packet `.claude/packets/TJ-11F.md` item 3. The trader, 2026-09-19: *"a veto on
friday night (after the market close) should not be considered monday since we
have new information then."*

The contract these tests pin, so the builder has nothing to guess
-----------------------------------------------------------------

``scripts/ui/services/day_review_service.py``
    ``_stamped_dates_for(session)`` = ``session`` plus the non-session dates
    that come AFTER it, up to but excluding the next session day. Friday
    2026-09-18 owns the Saturday and the Sunday; Monday 2026-09-21 owns only
    itself. Friday 2026-09-04 owns 09-05, 09-06 and Labor Day 09-07; Tuesday
    2026-09-08 owns only itself.

``scripts/walkaway_day.py``
    ``build(session, ...)`` keeps a decision when the session it was JUDGED on
    is ``session``: ``_row_session`` believes a stored ``decision_session``
    only when the row also carries ``decision_session_rule:
    "judged_session_v2"``, and otherwise recomputes from the row's own stamp
    through ``market_calendar.decision_session``. (Either the service tags its
    rows with that marker, or it lets the recompute answer - both land on the
    same session, because the service asked for exactly the dates that map
    onto it.)

    The D1 ruler follows the page, so a Friday-evening call is measured from
    **Friday's close**: a +10.0% run over the following week is +10.0% here,
    not the 0.0% it reads when Monday's gapped-up close is the reference.

``scripts/ui/annotations/store.py``
    ``load_annotations(..., by_decision_session=True)`` - TJ-11's opt-in, no
    production caller - answers the same mapping. Its DEFAULT path is exact
    `session_date` matching and **does not move**; the file is byte-identical
    after every read.

Plain dicts and ``tmp_path`` only: no live store, no clock, no network.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PACIFIC = ZoneInfo("America/Los_Angeles")

FRIDAY = "2026-09-18"
SATURDAY = "2026-09-19"
SUNDAY = "2026-09-20"
MONDAY = "2026-09-21"

#: The live stamp, to the microsecond: the first of the 18 D1 calls the trader
#: made after Friday's close on 2026-09-18.
FRIDAY_EVENING = "2026-09-18T21:04:28.734007-07:00"

#: Far enough past the five-session horizon that every window below is CLOSED,
#: so a pending clock can never be why a number is missing.
NOW = datetime(2026, 10, 5, 8, 0)

RULE_V2 = "judged_session_v2"


# -- bars --------------------------------------------------------------------


def _sessions_ending(session: str, count: int) -> list[str]:
    from market_calendar import previous_session

    cursor = date.fromisoformat(session)
    out = [cursor.isoformat()]
    for _ in range(count - 1):
        cursor = previous_session(cursor)
        out.append(cursor.isoformat())
    return list(reversed(out))


def _sessions_after(session: str, count: int) -> list[str]:
    from market_calendar import is_session

    cursor = date.fromisoformat(session)
    out: list[str] = []
    while len(out) < count:
        cursor += timedelta(days=1)
        if is_session(cursor):
            out.append(cursor.isoformat())
    return out


def _bar(day: str, o: float, h: float, low: float, c: float) -> dict:
    return {"dt": f"{day}T00:00:00", "open": o, "high": h, "low": low, "close": c}


def _gap_up_after_friday() -> list[dict]:
    """Flat to Friday's 100.00 close, then a week that never leaves 110.00.

    Fifteen flat bars each spanning exactly 2.00 make Wilder ATR(14) exactly
    2.00, so one ATR is exactly 2% and the arithmetic below is exact.

    Measured from FRIDAY's close the week ran +10.0% (5.00 ATR) with nothing
    against it. Measured from MONDAY's close - which is what the page does when
    the call is filed against Monday - the same week ran +0.0%. One set of
    bars, two answers, and only one of them is the judgement the trader made.
    """
    rows = [_bar(day, 100.0, 101.0, 99.0, 100.0) for day in _sessions_ending(FRIDAY, 15)]
    after = _sessions_after(FRIDAY, 5)
    rows.append(_bar(after[0], 110.0, 110.0, 100.0, 110.0))
    rows.extend(_bar(day, 110.0, 110.0, 110.0, 110.0) for day in after[1:])
    return rows


def _decision(**overrides) -> dict:
    """One decision row in the shape `day_review_service` hands the builder."""
    row = {
        "session_date": SATURDAY,
        "symbol": "HLIT",
        "side": "LONG",
        "verdict": "veto",
        "source": "annotations",
        "timeframe": "D1",
        "stamp": FRIDAY_EVENING,
        "category": "chart_review",
        "reason": "extended",
    }
    row.update(overrides)
    return row


def _build(session: str, decision: dict):
    from walkaway_day import build

    return build(
        session,
        sources={
            "decisions": (decision,),
            "preference": (),
            "outcomes": (),
            "scan_rows": (),
            "earlier_decisions": (),
        },
        bars={},
        now=NOW,
        daily_bars={"HLIT": _gap_up_after_friday()},
    )


# -- which dates belong to a session -----------------------------------------


def test_a_session_owns_the_non_session_dates_that_come_after_it():
    """Friday owns the weekend it was followed by, not the one before it."""
    from ui.services.day_review_service import _stamped_dates_for

    friday = _stamped_dates_for(FRIDAY)

    assert friday[0] == FRIDAY, friday
    assert set(friday) == {FRIDAY, SATURDAY, SUNDAY}
    # And Monday owns only itself: the weekend is Friday's.
    assert set(_stamped_dates_for(MONDAY)) == {MONDAY}


def test_a_session_before_a_holiday_owns_the_holiday_too():
    """Labor Day 2026 is Monday 2026-09-07; 09-04 is the session before it."""
    from ui.services.day_review_service import _stamped_dates_for

    assert set(_stamped_dates_for("2026-09-04")) == {
        "2026-09-04", "2026-09-05", "2026-09-06", "2026-09-07",
    }
    assert set(_stamped_dates_for("2026-09-08")) == {"2026-09-08"}


# -- where the row lands, and what it is measured from -----------------------


def test_a_friday_evening_call_is_measured_from_fridays_close():
    """The live case, with a number that only one reference can produce.

    The week after Friday gapped to 110.00 and stayed there. From Friday's
    100.00 close that is +10.0% and 5.00 ATR; from Monday's 110.00 close it is
    +0.0%. The trader judged it on Friday's close, so it is +10.0%.
    """
    day = _build(FRIDAY, _decision())

    assert [row.symbol for row in day.rejected] == ["HLIT"]
    row = day.rejected[0]
    assert row.ran_after_pct == pytest.approx(10.0)
    assert row.ran_after_atr == pytest.approx(5.0)
    assert row.against_first_pct == pytest.approx(0.0)
    assert row.at_close_pct == pytest.approx(10.0)
    assert row.real_miss == "run"
    assert row.state == "measured", row.state


def test_a_friday_evening_call_is_not_mondays_decision():
    """Monday's scan is new information; the call does not carry to it."""
    day = _build(MONDAY, _decision())

    assert [row.symbol for row in day.rejected] == []


def test_a_row_written_under_the_forward_rule_is_recomputed_not_believed():
    """A wave-1 row says Monday and has no marker, so its stamp decides.

    Zero rows on the desk carry this today (measured read-only on a copy,
    1,198 rows, 0 with the key) - but the desk could have written some between
    wave 1 going live at 2026-09-19 16:03 PDT and this fix, and they are never
    rewritten.
    """
    stale = _decision(decision_session=MONDAY)

    assert [row.symbol for row in _build(FRIDAY, stale).rejected] == ["HLIT"]
    assert [row.symbol for row in _build(MONDAY, stale).rejected] == []


def test_a_row_stamped_with_the_v2_marker_is_believed():
    """A marked row states its own session and the builder takes it."""
    marked = _decision(decision_session=FRIDAY, decision_session_rule=RULE_V2)

    assert [row.symbol for row in _build(FRIDAY, marked).rejected] == ["HLIT"]
    assert [row.symbol for row in _build(MONDAY, marked).rejected] == []


# -- the opt-in annotation read ----------------------------------------------


def _annotation(symbol: str, *, session_date: str, created_at: str) -> dict:
    return {
        "schema_version": 1, "event_id": f"e-{symbol}", "event_type": "veto",
        "symbol": symbol, "side": "SHORT", "session_date": session_date,
        "timeframe": "D1", "created_at": created_at, "source": "chart_review",
        "reason_code": "extended", "vocab_version": 3,
    }


def test_the_opt_in_annotation_read_answers_the_judged_session(tmp_path):
    """And the DEFAULT path does not move: it is what the live desk joins on."""
    from ui.annotations import store

    target = tmp_path / "trader_annotations.jsonl"
    rows = [
        _annotation("EEE", session_date=SATURDAY, created_at=FRIDAY_EVENING),
        _annotation("GGG", session_date=FRIDAY, created_at="2026-09-18T13:00:00-04:00"),
        _annotation("FFF", session_date=MONDAY, created_at="2026-09-21T13:00:00-04:00"),
    ]
    target.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    before = target.read_bytes()

    judged = store.load_annotations(target, session_date=FRIDAY, by_decision_session=True)
    assert sorted(row["symbol"] for row in judged) == ["EEE", "GGG"]

    monday = store.load_annotations(target, session_date=MONDAY, by_decision_session=True)
    assert [row["symbol"] for row in monday] == ["FFF"]

    # UNCHANGED, and it stays that way: `pick_feedback`, `review_learning`, the
    # cohort graders and `daily_recap_reader` all join on `session_date` by
    # exact match, and moving this default hid 12 symbols on 2026-09-19.
    base = store.load_annotations(target, session_date=FRIDAY)
    assert [row["symbol"] for row in base] == ["GGG"]

    # A reader maps; it never rewrites.
    assert target.read_bytes() == before
