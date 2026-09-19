"""When the Trade Mentor asks - WISHLIST 10J, packet WS-TM item 1.

A prompt is a **slot**: a named instant on the trader's own wall clock with a
kind and an expiry. This module answers one question - *which slots does this
session have?* - and answers it PURELY: no Qt, no I/O, no clock read, no state.
Everything that can drift (a timer that fired late, a desk that restarted, a
clock the OS corrected) is the service's problem, and it can only be its problem
because the schedule itself is a function of the date.

Three rules decide the tuple.

* **The hourly M5 reads run 07:00 Pacific until before the close.** Exclusive:
  a read filed at the closing bell is a read of a tape that has stopped.
  The close is the session's ACTUAL close, so a half day ends the hourly window
  three hours early - `market_early_close.session_close`, never
  `market_calendar.session_close`, which models every day as 16:00 ET by design.
* **The two D1 hours and the trade-check hour are FIXED and survive the close.**
  The trader asked for the noon D1 read on a short day in as many words, and the
  09:00 check asks about YESTERDAY's trades, not about today's tape. A slot at or
  after the close is kept and LABELLED `post_close`, so a later reader can tell a
  read of a live tape from a read of a closed one without re-deriving the
  calendar.
* **A collision is ONE slot, never two.** 08:00 is both an hourly M5 read and a
  D1 read; it produces a single `m5_d1` card; 09:00 is both an hourly read and
  the trade check and produces a single `m5_trades` card. Two cards at one
  instant is two dialogs over the chart, which the brief forbids.

Pacific is `America/Los_Angeles` - a wall clock WITH daylight saving, not a
fixed UTC-8. The 09:00 read is 09:00 in the trader's kitchen in March and in
November; a fixed offset would file half the year's reads an hour away from the
tape they describe.

Weekend, holiday, or a date outside the calendar's validated horizon: no slots
at all. Uncertainty asks nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

#: The trader's wall clock. DST-aware by construction.
PACIFIC = ZoneInfo("America/Los_Angeles")

#: First hourly M5 read of the session, Pacific.
FIRST_HOUR = 7
#: The two D1 reads. Fixed: they are kept even after an early close.
D1_HOURS = (8, 12)
#: The previous session's trade check. Fixed for the same reason.
#:
#: TJ-9 (trader, 2026-09-19: *"I want to be forced to label my trades around
#: 0900 as per trade mentor"*) moved this from 10 to 9. The hour is the whole
#: change: `_kind_for` follows the constant, so 10:00 becomes an ordinary `m5`
#: read and a short day no longer has a 10:00 slot forced into existence at
#: all - on 2026-11-27 (a 10:00 Pacific close) the session now carries four
#: slots rather than five, and the check sits INSIDE the hourly window instead
#: of being labelled `post_close`.
TRADES_HOUR = 9

#: An hourly read of the tape.
KIND_M5 = "m5"
#: An hourly read PLUS the daily read, on one card, stored as two entries.
KIND_M5_D1 = "m5_d1"
#: An hourly read PLUS the previous session's missing-field check.
KIND_M5_TRADES = "m5_trades"
#: Not scheduled at all - the trader pressed "Give a read". Never produced by
#: `slots_for_session`; the card builds one so a manual read carries the same
#: identity shape as a scheduled one.
KIND_MANUAL = "manual"

#: An unanswered prompt lives exactly one hour. On a normal session that instant
#: IS the next slot, which is why a backlog can never form; on a short day it
#: still never leaves two cards open at once.
SLOT_LIFETIME = timedelta(hours=1)


@dataclass(frozen=True)
class MentorSlot:
    """One scheduled prompt.

    `scheduled_at` and `expires_at` are aware Pacific instants; `session` is the
    exchange session the prompt belongs to, as an ISO date. `post_close` says
    the slot sits at or after that session's real close - it is a LABEL, never a
    reason to drop the slot.
    """

    slot_id: str
    session: str
    scheduled_at: datetime
    kind: str
    expires_at: datetime
    post_close: bool = False


def _kind_for(hour: int) -> str:
    if hour in D1_HOURS:
        return KIND_M5_D1
    if hour == TRADES_HOUR:
        return KIND_M5_TRADES
    return KIND_M5


def slot_id_for(session: str, moment: datetime, kind: str) -> str:
    """`<session>-<HHMM>-<kind>`; the identity the state file keys on."""
    return f"{session}-{moment.strftime('%H%M')}-{kind}"


def slots_for_session(session_date: date) -> tuple[MentorSlot, ...]:
    """Every prompt `session_date` carries, in time order.

    Empty for a weekend, a holiday, or a date the calendar refuses to
    extrapolate to. Never raises: a scheduler that cannot read the calendar
    prompts nothing, which is the safe direction for a feature whose whole job
    is to interrupt someone.
    """
    if isinstance(session_date, datetime):
        session_date = session_date.date()
    try:
        from market_calendar import is_session
        from market_early_close import session_close

        if not is_session(session_date):
            return ()
        close_pacific = session_close(session_date).astimezone(PACIFIC)
    except Exception:  # noqa: BLE001 - an unreadable calendar asks nothing
        return ()

    session = session_date.isoformat()
    hours = {hour for hour in range(FIRST_HOUR, close_pacific.hour)}
    hours.update(D1_HOURS)
    hours.add(TRADES_HOUR)

    slots: list[MentorSlot] = []
    for hour in sorted(hours):
        scheduled_at = datetime(
            session_date.year, session_date.month, session_date.day, hour, tzinfo=PACIFIC
        )
        kind = _kind_for(hour)
        slots.append(
            MentorSlot(
                slot_id=slot_id_for(session, scheduled_at, kind),
                session=session,
                scheduled_at=scheduled_at,
                kind=kind,
                expires_at=scheduled_at + SLOT_LIFETIME,
                post_close=scheduled_at >= close_pacific,
            )
        )
    return tuple(slots)


def manual_slot(moment: datetime) -> MentorSlot:
    """The identity a "Give a read" answer carries.

    A manual read is a real observation at a real time, so it gets the same slot
    shape as a scheduled one - and a kind that says nobody asked for it, so a
    later reader never counts it as an answered prompt.
    """
    local = moment.astimezone(PACIFIC)
    session = local.date().isoformat()
    return MentorSlot(
        slot_id=slot_id_for(session, local, KIND_MANUAL),
        session=session,
        scheduled_at=local,
        kind=KIND_MANUAL,
        expires_at=local + SLOT_LIFETIME,
        post_close=False,
    )
