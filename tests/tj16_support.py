"""Hand-built fixtures for the TJ-16 red tests. NOT a test module.

Nothing here is produced by the code TJ-16 will build. The journal entries go
through `market_journal.build_entry` / `build_prediction`, the read rows through
`market_read_grades.read_rows` and the verdicts through
`market_read_grades.grade_read` - all four shipped with TJ-10 and TJ-14A and
none of them is under test in this packet - so a fixture row is the row the desk
would actually have written, not a hand-typed dict in the shape somebody hoped
for.

THE TAPE
--------
One benchmark tape per session, 78 completed five-minute bars from 06:30 to
12:55 Pacific, every bar flat (``open == close``) with a half-point wick::

    bars  0 .. 6   level  99.0 (rising) / 101.0 (falling)
    bars  7 .. 76  level 100.0
    bar   77       level 102.0 (rising) /  98.0 (falling)

Every read is stamped at ``hh:02`` Pacific, inside a bar, so the first completed
bar that STARTS after the stamp always opens at 100.0 and the session's last
completed bar always closes at 102.0 or 98.0. The move is therefore EXACTLY
+2.00 on a rising session and -2.00 on a falling one, whatever the hour - so a
row's verdict is decided by its own DIRECTION and nothing else, and the counts
below are countable by hand.

THE BAND
--------
The flat band is `market_read_grades.FLAT_BAND_ATR`, the builder's constant, and
no number here restates it: :func:`atr_for` asks for the ATR that puts a 2.00
move at a chosen MULTIPLE of the band. Three band-widths is a real move (`up`
on a rising session is `right`); half a band-width is `flat`.

THE LEDGER
----------
:func:`two_weeks_of_clicks` is ten exchange sessions of four hourly clicks -
forty rest-of-day rows - and every count the tests assert is written out in its
docstring. :func:`calibration_ledger` is twenty-four sessions of four, sized so
each `How sure` bucket clears `evidence_stats.MIN_REPORTABLE_N` on its own.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PACIFIC = ZoneInfo("America/Los_Angeles")

#: The session the whole TJ-16 fixture ends on: a Friday, a real regular-close
#: exchange session (the same one TJ-10's fixtures use).
LAST_SESSION = "2026-09-18"

#: 06:30 Pacific = 09:30 New York.
OPEN_HOUR, OPEN_MINUTE = 6, 30
BAR_COUNT = 78

#: The four hours a session's clicks are stamped at, Pacific. All four sit
#: inside the flat middle of the tape.
CLICK_HOURS = (7, 8, 9, 10)

BEFORE_RISING, BEFORE_FALLING = 99.0, 101.0
ANCHOR_LEVEL = 100.0
CLOSE_RISING, CLOSE_FALLING = 102.0, 98.0

#: What the tape actually moves, in price, from the anchor to the close.
SESSION_MOVE = 2.0


def atr_for(move: float, band_multiple: float) -> float:
    """The ATR that puts ``move`` at ``band_multiple`` times the module's band."""
    from market_read_grades import FLAT_BAND_ATR

    return abs(move) / (band_multiple * FLAT_BAND_ATR)


def sessions_ending(session: str = LAST_SESSION, count: int = 10) -> list[str]:
    """``count`` real exchange sessions ending at ``session``, oldest first."""
    from market_calendar import previous_session

    cursor = date.fromisoformat(session)
    out = [cursor.isoformat()]
    while len(out) < count:
        cursor = previous_session(cursor)
        out.append(cursor.isoformat())
    return list(reversed(out))


def _bar(day: date, index: int, level: float) -> dict[str, Any]:
    start = datetime(
        day.year, day.month, day.day, OPEN_HOUR, OPEN_MINUTE, tzinfo=PACIFIC
    ) + timedelta(minutes=5 * index)
    return {
        "dt": start,
        "open": level,
        "high": level + 0.5,
        "low": level - 0.5,
        "close": level,
        "volume": 1_000,
    }


def session_tape(session: str, *, rising: bool) -> list[dict[str, Any]]:
    """One session's benchmark tape. See the module docstring for its shape."""
    day = date.fromisoformat(session)
    before = BEFORE_RISING if rising else BEFORE_FALLING
    last = CLOSE_RISING if rising else CLOSE_FALLING
    levels = [before] * 7 + [ANCHOR_LEVEL] * 70 + [last]
    return [_bar(day, index, levels[index]) for index in range(BAR_COUNT)]


def daily_ramp(sessions: Sequence[str], *, start: float = 100.0, step: float = 1.0):
    """Daily bars, one per session, closing a point higher each time."""
    return [
        {
            "dt": f"{day}T00:00:00",
            "open": start + step * index,
            "high": start + step * index + 1.0,
            "low": start + step * index - 1.0,
            "close": start + step * index,
            "volume": 1_000_000,
        }
        for index, day in enumerate(sessions)
    ]


def stamp_at(session: str, hour: int, minute: int = 2) -> datetime:
    day = date.fromisoformat(session)
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=PACIFIC)


def morning_after(session: str) -> datetime:
    """08:00 Pacific the next calendar day - every horizon of that day is shut."""
    day = date.fromisoformat(session) + timedelta(days=1)
    return datetime(day.year, day.month, day.day, 8, 0, tzinfo=PACIFIC)


# ---------------------------------------------------------------------------
# entries -> read rows -> graded rows, through the shipped TJ-10 path
# ---------------------------------------------------------------------------
def click_entry(
    *,
    session: str,
    hour: int,
    direction: str,
    confidence: str,
    horizon: str = "rest_of_day",
    observation: str = "",
    because: str = "",
) -> dict[str, Any]:
    """One REAL journal row carrying one forced prediction click."""
    import market_journal

    moment = stamp_at(session, hour)
    timeframe = (
        market_journal.TIMEFRAME_D1
        if horizon == market_journal.HORIZON_NEXT_5_SESSIONS
        else market_journal.TIMEFRAME_M5
    )
    text = observation or f"The tape at {hour:02d}00 on {session}."
    mentor = {
        "slot_id": f"{timeframe.lower()}-{hour:02d}00",
        "prompt_kind": timeframe.lower(),
        "scheduled_at": (moment - timedelta(minutes=2)).isoformat(),
        "responded_at": moment.isoformat(),
        "observation": text,
        "prediction": market_journal.build_prediction(
            direction=direction,
            horizon=horizon,
            confidence=confidence,
            because=because,
        ),
    }
    return market_journal.build_entry(
        text=text,
        session_date=session,
        timeframe=timeframe,
        origin=market_journal.ORIGIN_TRADE_MENTOR,
        mentor=mentor,
        now=moment,
    )


def context(
    *,
    hour: int,
    direction: str,
    confidence: str,
    last_hour_spy: str,
    d1_environment: str,
    spy_vs_prior_range: str = "unmeasured",
    gap_pct: Any = "unmeasured",
    previous_call_verdict: str = "none",
    **extra: Any,
) -> dict[str, Any]:
    """A point-in-time snapshot in `market_read_grades.context_for`'s own shape.

    Written by hand rather than measured, so a test can put a chosen value in a
    chosen field. Every key `context_for` returns is present - an old row has
    its keys PRESENT, and a field the desk could not measure reads `unmeasured`,
    never 0 and never absent.
    """
    block = {
        "internals": {"schema": "trade_mentor_context_v2", "readings": [], "derived": {}},
        "hour": hour,
        "spy_vs_session_vwap": "above",
        "spy_vs_prior_range": spy_vs_prior_range,
        "gap_pct": gap_pct,
        "d1_environment": d1_environment,
        "last_hour_spy": last_hour_spy,
        "agrees_with_own_d1": "unmeasured",
        "previous_call_verdict": previous_call_verdict,
        "confidence": confidence,
        "direction": direction,
    }
    block.update(extra)
    return block


def graded_session(
    session: str,
    specs: Sequence[Mapping[str, Any]],
    *,
    rising: bool,
    band_multiple: float = 3.0,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """One session's clicks, graded by the REAL grader against the real tape.

    ``specs`` is one mapping per click: ``hour``, ``direction``, ``confidence``
    and whatever context overrides the test wants. The verdict is decided by
    `market_read_grades.grade_read` and by the tape, never assigned here.
    """
    import market_read_grades as grades

    moment = now or morning_after(session)
    tape = session_tape(session, rising=rising)
    atr = atr_for(SESSION_MOVE, band_multiple)
    entries = [
        click_entry(
            session=session,
            hour=int(spec["hour"]),
            direction=str(spec["direction"]),
            confidence=str(spec.get("confidence") or "medium"),
            observation=str(spec.get("observation") or ""),
            because=str(spec.get("because") or ""),
        )
        for spec in specs
    ]
    rows = grades.read_rows(entries, session=session)
    assert len(rows) == len(specs), (session, len(rows), len(specs))
    out = []
    for spec, row in zip(specs, rows):
        overrides = {
            key: value
            for key, value in spec.items()
            if key not in (
                "hour", "direction", "confidence", "observation", "because",
                "last_hour_spy", "d1_environment",
            )
        }
        out.append(
            grades.grade_read(
                row,
                m5_bars=tape,
                atr=atr,
                now=moment,
                context=context(
                    hour=int(spec["hour"]),
                    direction=str(spec["direction"]),
                    confidence=str(spec.get("confidence") or "medium"),
                    last_hour_spy=str(spec.get("last_hour_spy") or "up"),
                    d1_environment=str(
                        spec.get("d1_environment")
                        or ("trending_up" if rising else "trending_down")
                    ),
                    **overrides,
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# the two ledgers the tests read
# ---------------------------------------------------------------------------
#: Ten sessions, four clicks an hour apart, by the hand-counted plan below.
#:
#: Sessions 0-5 RISE (+2.00), sessions 6-9 FALL (-2.00). The directions:
#:
#:   s0, s1, s2 (rising)  up   up   up   up     -> 12 right,  0 wrong
#:   s3, s4, s5 (rising)  up   up   down down   ->  6 right,  6 wrong
#:   s6         (falling) down down down down   ->  4 right,  0 wrong
#:   s7, s8, s9 (falling) up   up   up   up     ->  0 right, 12 wrong
#:
#: TOTAL: 40 closed rows, **22 right and 18 wrong**, so the trader's rate is
#: 22/40 = 0.55. The three naive rules answer the SAME forty stamps:
#:
#:   always_up                 right on every rising row      24/40 = 0.60
#:   same_as_the_last_hour     the last hour lied on s2, s8   32/40 = 0.80
#:   with_the_d1_environment   the label lied on s4, s9       32/40 = 0.80
#:
#: Every one of them beats the trader, which is the sentence item 2 exists to
#: make possible - and none of them is a perfect 40/40, because a baseline that
#: never misses is a fixture, not a comparison.
#:
#: `How sure` is assigned by session so each bucket's arithmetic is countable:
#: s0-s2 `low` (12 right of 12), s3-s5 `medium` (6 of 12), s6-s9 `high` (4 of
#: 16). Low beats high, and a page that claims otherwise is lying.
_RISING_ALL_UP = ("up", "up", "up", "up")
_RISING_SPLIT = ("up", "up", "down", "down")
_FALLING_ALL_DOWN = ("down", "down", "down", "down")
_FALLING_ALL_UP = ("up", "up", "up", "up")

TWO_WEEKS_PLAN = (
    # (rising, directions, confidence, last_hour_spy, d1_environment)
    (True, _RISING_ALL_UP, "low", "up", "trending_up"),
    (True, _RISING_ALL_UP, "low", "up", "trending_up"),
    (True, _RISING_ALL_UP, "low", "down", "trending_up"),
    (True, _RISING_SPLIT, "medium", "up", "trending_up"),
    (True, _RISING_SPLIT, "medium", "up", "trending_down"),
    (True, _RISING_SPLIT, "medium", "up", "trending_up"),
    (False, _FALLING_ALL_DOWN, "high", "down", "trending_down"),
    (False, _FALLING_ALL_UP, "high", "down", "trending_down"),
    (False, _FALLING_ALL_UP, "high", "up", "trending_down"),
    (False, _FALLING_ALL_UP, "high", "down", "trending_up"),
)

#: Which of the forty rows carry a MEASURED `spy_vs_prior_range`. The first two
#: sessions' rows read `unmeasured` (8 rows: 4 right in s0, 4 right in s1 - no,
#: see below), so the feature's population is 32, not 40.
#:
#: s0 (4 right) and s7 (4 wrong) are the unmeasured pair: 40 - 8 = 32 rows
#: measured, of which 22-4 = 18 are right and 18-4 = 14 are wrong. Both sides
#: clear `evidence_contrast.MIN_CONTRAST_SIDE_N` (10) and 32 clears
#: `MIN_REPORTABLE_N` (30), so the feature is RANKED on 32 rows - and a builder
#: that read the eight blanks as zeros would rank it on 40.
UNMEASURED_RANGE_SESSIONS = (0, 7)

#: Which rows carry a measured `gap_pct`: five rows only (three right, two
#: wrong), far under the feature floor, so it is NAMED in `thin_features` with
#: its two counts and never ranked.
THIN_GAP_ROWS = ((0, 0), (1, 0), (2, 0), (7, 0), (8, 0))


def two_weeks_of_clicks(
    *, last_session: str = LAST_SESSION
) -> tuple[list[str], list[dict[str, Any]]]:
    """``(sessions, grades)`` - ten sessions, forty graded rest-of-day rows."""
    sessions = sessions_ending(last_session, len(TWO_WEEKS_PLAN))
    out: list[dict[str, Any]] = []
    for index, (rising, directions, confidence, last_hour, label) in enumerate(
        TWO_WEEKS_PLAN
    ):
        specs = []
        for position, hour in enumerate(CLICK_HOURS):
            spec: dict[str, Any] = {
                "hour": hour,
                "direction": directions[position],
                "confidence": confidence,
                # What the two reading baselines have to read. Neither is
                # simply "whatever the session did": see the plan above.
                "last_hour_spy": last_hour,
                "d1_environment": label,
                "spy_vs_prior_range": (
                    "unmeasured"
                    if index in UNMEASURED_RANGE_SESSIONS
                    else ("above" if rising else "below")
                ),
                "gap_pct": 0.4 if (index, position) in THIN_GAP_ROWS else "unmeasured",
            }
            specs.append(spec)
        out.extend(graded_session(sessions[index], specs, rising=rising))
    return sessions, out


#: Twenty-four sessions of four clicks, one `How sure` per session, sized so
#: every bucket clears `MIN_REPORTABLE_N` on its own:
#:
#:   low     8 sessions - 7 all-right, 1 all-wrong  -> 28 of 32 = 0.875
#:   medium  8 sessions - 4 all-right, 4 all-wrong  -> 16 of 32 = 0.50
#:   high    8 sessions - 2 all-right, 6 all-wrong  ->  8 of 32 = 0.25
#:
#: 96 rows, 52 right. **Low beats high**, so a page that prints "you are right
#: more often when you are sure" is printing the opposite of its own numbers.
CALIBRATION_PLAN = tuple(
    [("low", True)] * 7 + [("low", False)]
    + [("medium", True)] * 4 + [("medium", False)] * 4
    + [("high", True)] * 2 + [("high", False)] * 6
)


def calibration_ledger(
    *, last_session: str = LAST_SESSION
) -> tuple[list[str], list[dict[str, Any]]]:
    """``(sessions, grades)`` - 96 graded rows, 32 per `How sure` bucket."""
    sessions = sessions_ending(last_session, len(CALIBRATION_PLAN))
    out: list[dict[str, Any]] = []
    for index, (confidence, correct) in enumerate(CALIBRATION_PLAN):
        # Every tape rises; a session is "all right" when the trader called up
        # and "all wrong" when they called down.
        direction = "up" if correct else "down"
        specs = [
            {"hour": hour, "direction": direction, "confidence": confidence}
            for hour in CLICK_HOURS
        ]
        out.extend(graded_session(sessions[index], specs, rising=True))
    return sessions, out


def five_session_clicks(
    sessions: Sequence[str], *, count: int = 5
) -> list[dict[str, Any]]:
    """``count`` graded `next_5_sessions` rows, the oldest sessions first.

    Three are `up` and two are `down` against a daily ramp that rises a point a
    session, so three are right and two are wrong on a 5.00 move.
    """
    import market_read_grades as grades

    bars = daily_ramp(sessions)
    atr = atr_for(5.0, 3.0)
    moment = morning_after(sessions[-1])
    out: list[dict[str, Any]] = []
    for index in range(count):
        session = sessions[index]
        direction = "up" if index < 3 else "down"
        entry = click_entry(
            session=session,
            hour=8,
            direction=direction,
            confidence="medium",
            horizon="next_5_sessions",
            observation=f"The weekly picture on {session}.",
        )
        rows = grades.read_rows([entry], session=session)
        assert len(rows) == 1, rows
        out.append(
            grades.grade_read(
                rows[0],
                daily_bars=bars,
                atr=atr,
                now=moment,
                context=context(
                    hour=8,
                    direction=direction,
                    confidence="medium",
                    last_hour_spy="up",
                    d1_environment="trending_up",
                ),
            )
        )
    return out


def write_ledger(root: Path, grades_by_session: Mapping[str, Sequence[Mapping[str, Any]]]):
    """Append every row through the store's OWN writer, refusals and all."""
    import market_read_grades as grades

    for session, rows in grades_by_session.items():
        grades.append_grades(session, list(rows), root=root)
    return root


def by_session(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get("session") or ""), []).append(dict(row))
    return out


def store_ledger(root: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    write_ledger(root, by_session(rows))
    return root
