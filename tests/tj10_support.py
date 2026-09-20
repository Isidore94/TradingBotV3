"""Hand-built fixtures for the TJ-10 red tests. NOT a test module.

Nothing here is produced by the code under test. Every bar is written from a
level chosen by hand, and every expected number in the tests is derived from
those levels by arithmetic written out in the test itself.

THE SESSION
-----------
Friday **2026-09-18**, a plain regular-close session (16:00 New York = 13:00
Pacific), which is a real session on the desk's own calendar. Its next five
exchange sessions are 09-21, 09-22, 09-23, 09-24 and 09-25, so the D1 horizon's
checkpoints at 1, 3 and 5 sessions are 09-21, 09-23 and 09-25.

The durable session tape (`day_review_bars`) is PACIFIC-local, so every bar
here is too: 78 completed five-minute bars from 06:30 to 12:55 Pacific.

THE TAPE'S SHAPE, chosen so one wrong anchor is a different number
------------------------------------------------------------------
Every bar is flat (``open == close``) with a half-point wick either side::

    bars  0 .. 6   level  99.0     <- at or BEFORE the 07:02 stamp
    bars  7 .. 76  level 100.0     <- the first bar AFTER the stamp opens here
    bar   77       level 102.0     <- the session's last completed bar

So the only honest anchor is ``100.0`` (bar 7's open, the first completed bar
that STARTS after the stamp) and the only honest close is ``102.0``: a grader
that anchored on the stamp's own bar would read 99.0 and a move of +3.00, and a
grader that took the session's first bar would read the same.

THE ATR IS DERIVED FROM THE MODULE'S OWN BAND
---------------------------------------------
The flat band is a constant the builder declares (`plan.md` TJ-10 item 2: "the
band is ONE constant in ATR with its reason written beside it") and this packet
states no number, so no test here asserts one. Instead each test asks for an
ATR that puts the measured move at a chosen MULTIPLE of whatever the band is -
:func:`atr_for` - so "three band-widths up is `right`" and "half a band-width
is `flat`" are true of every band the builder could pick, and a grader that
ignores the band fails either way.
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
EASTERN = ZoneInfo("America/New_York")

#: A Friday, and a real regular-close session.
SESSION = "2026-09-18"
SESSION_DATE = date(2026, 9, 18)
#: The five exchange sessions after it, verified against `market_calendar`.
NEXT_SESSIONS = ("2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25")

#: 06:30 Pacific = 09:30 New York.
FIRST_BAR = datetime(2026, 9, 18, 6, 30, tzinfo=PACIFIC)
#: 78 five-minute bars: the last one starts 12:55 and closes at the 13:00 bell.
BAR_COUNT = 78

#: The trader answered the card at 07:02 Pacific - INSIDE bar 6 (07:00-07:05),
#: so the first completed bar after the stamp is bar 7.
STAMP = datetime(2026, 9, 18, 7, 2, tzinfo=PACIFIC)
STAMP_BAR = 6
ANCHOR_BAR = 7

BEFORE_LEVEL = 99.0
ANCHOR_LEVEL = 100.0
CLOSE_LEVEL = 102.0

#: The move the tape below actually makes, in price. Written out rather than
#: measured, so the test states the answer and the code has to reach it.
SESSION_MOVE = CLOSE_LEVEL - ANCHOR_LEVEL  # +2.00

#: The morning after, so the whole session is closed and complete.
AFTER_THE_CLOSE = datetime(2026, 9, 19, 8, 0, tzinfo=PACIFIC)
#: Two minutes before the bell, so bar 77 is still FORMING.
BEFORE_THE_BELL = datetime(2026, 9, 18, 12, 58, tzinfo=PACIFIC)


def _bar(index: int, level: float, *, start: datetime = FIRST_BAR) -> dict[str, Any]:
    return {
        "dt": start + timedelta(minutes=5 * index),
        "open": level,
        "high": level + 0.5,
        "low": level - 0.5,
        "close": level,
        "volume": 1_000,
    }


def session_tape(*, through: int = BAR_COUNT) -> list[dict[str, Any]]:
    """The benchmark's Pacific-local M5 tape, `through` bars of it."""
    levels = (
        [BEFORE_LEVEL] * 7
        + [ANCHOR_LEVEL] * 70
        + [CLOSE_LEVEL]
    )
    return [_bar(index, levels[index]) for index in range(min(through, BAR_COUNT))]


def atr_for(move: float, band_multiple: float) -> float:
    """The ATR that puts `move` at `band_multiple` times the module's band.

    The band's VALUE is the builder's to declare; its MEANING is the packet's.
    """
    from market_read_grades import FLAT_BAND_ATR

    return abs(move) / (band_multiple * FLAT_BAND_ATR)


def daily_bars(closes: Mapping[str, float]) -> list[dict[str, Any]]:
    """Daily bars, oldest first, one per named session date."""
    return [
        {
            "dt": day,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000,
        }
        for day, close in closes.items()
    ]


def mentor_entry(
    *,
    direction: str,
    horizon: str,
    timeframe: str,
    text: str = "",
    confidence: str = "medium",
    because: str = "",
    responded_at: datetime | None = None,
    created_at: datetime | None = None,
    session: str = SESSION,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One REAL journal row with a clicked prediction, through the real writer."""
    import market_journal

    moment = created_at or responded_at or STAMP
    mentor: dict[str, Any] = {
        "slot_id": "m5-0700",
        "prompt_kind": "m5",
        "scheduled_at": (moment - timedelta(minutes=2)).isoformat(),
        "responded_at": (responded_at or moment).isoformat(),
        "observation": text,
        "prediction": market_journal.build_prediction(
            direction=direction,
            horizon=horizon,
            confidence=confidence,
            because=because,
        ),
    }
    if context is not None:
        mentor["context"] = dict(context)
    return market_journal.build_entry(
        text=text,
        session_date=session,
        timeframe=timeframe,
        origin=market_journal.ORIGIN_TRADE_MENTOR,
        mentor=mentor,
        now=moment,
    )


#: The FOUR mentor vintages the live September ledger actually holds, measured
#: on a copy 2026-09-19: no `mentor` key at all (43 rows), `mentor == {}` (13),
#: a mentor payload with no context (6) and a full v1 context (22). None of them
#: is a click, and `prediction_of` answers `None` for every one - so each is an
#: EXTRACTED read or no read at all, never a clicked one. An old row's key is
#: PRESENT and EMPTY, not absent, which is why all four are modelled.
def old_entry(
    vintage: str,
    *,
    text: str,
    created_at: datetime | None = None,
    timeframe: str = "M5",
    session: str = SESSION,
) -> dict[str, Any]:
    import market_journal

    moment = created_at or STAMP
    mentor: Any
    if vintage == "absent":
        mentor = None
    elif vintage == "empty":
        mentor = {}
    elif vintage == "no_context":
        mentor = {"slot_id": "m5-0700", "prompt_kind": "m5",
                  "responded_at": moment.isoformat()}
    elif vintage == "context_v1":
        mentor = {
            "slot_id": "m5-0700", "prompt_kind": "m5",
            "responded_at": moment.isoformat(),
            "context": {"schema": "trade_mentor_context_v1", "readings": []},
        }
    else:  # pragma: no cover - a typo in a test is a test defect
        raise ValueError(f"no such vintage: {vintage!r}")
    row = market_journal.build_entry(
        text=text,
        session_date=session,
        timeframe=timeframe,
        origin=market_journal.ORIGIN_TRADE_MENTOR,
        mentor=mentor if isinstance(mentor, dict) else None,
        now=moment,
    )
    if vintage == "absent":
        # `build_entry` writes `mentor: {}` on every row it builds today, so the
        # oldest live vintage - which has no `mentor` KEY at all - can only be
        # modelled by removing it. 43 of the live rows look exactly like this.
        row.pop("mentor", None)
    return row


def live_notes() -> list[dict[str, Any]]:
    """The 49 live note texts, with the stance the OLD extractor read."""
    import json

    payload = json.loads(
        (ROOT_DIR / "tests" / "fixtures" / "tj10_live_notes.json").read_text(
            encoding="utf-8"
        )
    )
    return list(payload["notes"])


def note_entry(note: Mapping[str, Any]) -> dict[str, Any]:
    """One fixture note as a journal-shaped row. TEXT and id only."""
    return {
        "event_type": "entry",
        "entry_id": note["entry_id"],
        "session_date": SESSION,
        "created_at": STAMP.astimezone(ZoneInfo("UTC")).isoformat(timespec="seconds"),
        "timeframe": note.get("timeframe") or "M5",
        "symbols": [],
        "origin": "trade_mentor",
        "text": note["text"],
        "written_after_the_session": bool(note.get("written_after_the_session")),
        "mentor": {},
        "supersedes": "",
        "reaffirms": "",
    }


def stamps_of(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("stamp") or "") for row in rows]
