"""Packet WS-TM - Trade Mentor: the schedule, the service, the card, the 10:00 task.

WISHLIST 10J steps 1-2. These tests were written BEFORE the feature and are red on
`claude/ws-tm-trade-mentor`'s base commit; the builder makes them pass and may only ADD.

WHAT THEY PIN, AND THE NAMES THEY PIN IT TO
-------------------------------------------
The packet names three modules and leaves a fourth unnamed. The names below are the
contract the builder inherits; nothing here may be renamed away to make a test pass.

``scripts/trade_mentor_schedule.py`` (pure, no Qt, no I/O)
    ``FIRST_HOUR = 7``, ``D1_HOURS = (8, 12)``, ``TRADES_HOUR = 10``,
    ``PACIFIC`` (``ZoneInfo("America/Los_Angeles")``),
    ``KIND_M5 = "m5"`` / ``KIND_M5_D1 = "m5_d1"`` / ``KIND_M5_TRADES = "m5_trades"``,
    ``MentorSlot(slot_id, session, scheduled_at, kind, expires_at, post_close)`` and
    ``slots_for_session(session_date) -> tuple[MentorSlot, ...]``.

``scripts/ui/services/trade_mentor_service.py``
    ``IDLE_GRACE_MINUTES = 20``, ``TradeMentorService(clock=, idle_seconds=,
    session_locked=, state_path=)`` with ``poll()``, ``pause_today()``,
    ``slot_state(slot_id)``, ``next_prompt_at(now)`` and the two signals
    ``promptDue(object)`` / ``promptExpired(str)``.

``scripts/ui/widgets/trade_mentor_card.py``
    ``TradeMentorCard(parent=None, *, journal=, clock=, drafts_path=)`` with
    ``show_slot(slot, previous=None)``, ``text_box``, ``submit()``,
    ``read_unchanged()``, ``skip()``, ``give_a_read(now=None)``, ``draft_for(slot_id)``.

``scripts/trade_mentor_trade_check.py`` (the packet describes item 4 but names no file)
    the four answer states, ``MATERIAL_FIELDS``, ``TRADE_CAP_DEFAULT = 3``,
    ``build_task(store, session, cap=)``, ``save_answers(...)``, ``recalled_fields(...)``.

WHY EACH ASSERTION IS A NUMBER AND NOT A SHAPE
----------------------------------------------
* The schedule is asserted as the exact tuple of ``(HH:MM, kind, post_close)`` for a
  normal day (six slots), an early close (five, two of them post-close) and a holiday
  (none). A formula that emits an M5 read after the close, or drops the noon D1 read on
  a short day, produces a different tuple and fails.
* DST is asserted as the UTC OFFSET of the same wall hour on two sessions that sit on
  opposite sides of a transition (-08:00 on 2026-03-06, -07:00 on 2026-03-09). A fixed
  UTC-8 implementation passes the first and fails the second. The two transition dates
  the packet names, 2026-03-08 and 2026-11-01, are both SUNDAYS, so they are asserted
  empty and the offset proof is carried by their neighbouring sessions.
* Presence is driven through its real inputs: the autopilot state FILE for AWAY, the
  service's own ``pause_today()`` for paused, and injected ``idle_seconds`` /
  ``session_locked`` callables for the two primitives that read the OS.
* The journal write is read back out of a real ``EvidenceLedger`` in ``tmp_path``, so a
  "write" that never reached disk cannot pass.

Never sleeps: every clock is injected and every "an hour later" is a new call.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Trade Mentor card and service are Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_app = QApplication.instance() or QApplication([])

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")

#: A plain Monday. Regular close 16:00 ET = 13:00 PT.
NORMAL_SESSION = date(2026, 9, 14)
#: The day after Thanksgiving 2026: a real early close, 13:00 ET = 10:00 PT.
EARLY_CLOSE_SESSION = date(2026, 11, 27)
#: Thanksgiving itself: the market is shut.
HOLIDAY = date(2026, 11, 26)
#: The two DST transitions the packet names. Both are Sundays.
DST_SPRING_FORWARD = date(2026, 3, 8)
DST_FALL_BACK = date(2026, 11, 1)


def _pacific(day: date, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=PACIFIC_TZ)


def _shape(slots) -> tuple[tuple[str, str, bool], ...]:
    """The schedule as (wall time, kind, post-close) - the three things that matter."""
    return tuple(
        (slot.scheduled_at.strftime("%H:%M"), slot.kind, bool(slot.post_close))
        for slot in slots
    )


# ---------------------------------------------------------------------------
# Item 1 - the pure schedule
# ---------------------------------------------------------------------------


def test_a_normal_session_asks_every_whole_hour_from_seven_until_before_the_close():
    """07:00 to 12:00 Pacific: six prompts, with D1 folded into 08 and 12 and the
    previous session's trades folded into 10. The close is 13:00 PT, so there is no
    13:00 read - "until before the close" is exclusive."""
    import trade_mentor_schedule as schedule

    slots = schedule.slots_for_session(NORMAL_SESSION)

    assert _shape(slots) == (
        ("07:00", "m5", False),
        ("08:00", "m5_d1", False),
        ("09:00", "m5", False),
        ("10:00", "m5_trades", False),
        ("11:00", "m5", False),
        ("12:00", "m5_d1", False),
    )
    assert schedule.FIRST_HOUR == 7
    assert schedule.D1_HOURS == (8, 12)
    assert schedule.TRADES_HOUR == 10


def test_the_eight_ten_and_twelve_collisions_each_produce_one_combined_slot():
    """One card, not two. An hourly M5 read that lands on a D1 hour or on the trade
    hour is the SAME slot with a combined kind - two slots at 08:00 would stack two
    dialogs, which the trader's brief forbids."""
    import trade_mentor_schedule as schedule

    slots = schedule.slots_for_session(NORMAL_SESSION)
    by_hour: dict[int, list] = {}
    for slot in slots:
        by_hour.setdefault(slot.scheduled_at.hour, []).append(slot)

    assert [len(by_hour[hour]) for hour in (8, 10, 12)] == [1, 1, 1]
    assert by_hour[8][0].kind == schedule.KIND_M5_D1
    assert by_hour[10][0].kind == schedule.KIND_M5_TRADES
    assert by_hour[12][0].kind == schedule.KIND_M5_D1
    # And the combined ones are still M5 reads: the kind carries both halves.
    assert by_hour[8][0].kind.startswith("m5")


def test_an_early_close_keeps_the_fixed_slots_and_drops_the_hourly_reads_after_it():
    """2026-11-27 closes at 10:00 Pacific. The hourly M5 window ends before that, so
    there is no 11:00 read at all - but the trader explicitly asked to keep the noon
    D1 read, and the 10:00 trade check is about YESTERDAY, not about today's tape. Both
    survive, labelled post-close."""
    import trade_mentor_schedule as schedule

    slots = schedule.slots_for_session(EARLY_CLOSE_SESSION)

    assert _shape(slots) == (
        ("07:00", "m5", False),
        ("08:00", "m5_d1", False),
        ("09:00", "m5", False),
        ("10:00", "m5_trades", True),
        ("12:00", "m5_d1", True),
    )
    # The regular calendar would say 16:00 ET here; only the early-close calendar
    # knows better, and an implementation that asks the wrong one emits an 11:00 read.
    assert 11 not in {slot.scheduled_at.hour for slot in slots}


def test_a_holiday_and_a_weekend_have_no_prompts_at_all():
    import trade_mentor_schedule as schedule

    assert schedule.slots_for_session(HOLIDAY) == ()
    assert schedule.slots_for_session(DST_SPRING_FORWARD) == ()
    assert schedule.slots_for_session(DST_FALL_BACK) == ()


def test_pacific_wall_time_follows_daylight_saving_on_both_sides_of_a_transition():
    """"PST" means the wall clock in Los Angeles, not a fixed UTC-8. The 09:00 read on
    the Friday before the spring change is -08:00 and on the Monday after it is -07:00;
    a `timezone(timedelta(hours=-8))` implementation gets the second one wrong by an
    hour and files the read against the wrong point in the tape."""
    import trade_mentor_schedule as schedule

    def nine_am(day: date):
        for slot in schedule.slots_for_session(day):
            if slot.scheduled_at.hour == 9:
                return slot
        raise AssertionError(f"no 09:00 slot on {day}")

    before_spring = nine_am(date(2026, 3, 6))
    after_spring = nine_am(date(2026, 3, 9))
    before_fall = nine_am(date(2026, 10, 30))
    after_fall = nine_am(date(2026, 11, 2))

    assert before_spring.scheduled_at.utcoffset() == timedelta(hours=-8)
    assert after_spring.scheduled_at.utcoffset() == timedelta(hours=-7)
    assert before_fall.scheduled_at.utcoffset() == timedelta(hours=-7)
    assert after_fall.scheduled_at.utcoffset() == timedelta(hours=-8)
    assert str(after_spring.scheduled_at.tzinfo) == "America/Los_Angeles"


def test_a_slot_id_names_its_session_its_wall_time_and_its_kind():
    import trade_mentor_schedule as schedule

    slots = schedule.slots_for_session(NORMAL_SESSION)
    ids = [slot.slot_id for slot in slots]

    assert ids[0] == "2026-09-14-0700-m5"
    assert ids[3] == "2026-09-14-1000-m5_trades"
    assert len(set(ids)) == len(ids)
    assert all(slot.session == "2026-09-14" for slot in slots)


def test_an_unanswered_prompt_expires_one_hour_after_it_was_scheduled():
    """The trader's own words: "expire unanswered prompts" at the next hour. On a
    normal day that instant IS the next slot, which is why a backlog can never form."""
    import trade_mentor_schedule as schedule

    slots = schedule.slots_for_session(NORMAL_SESSION)

    for slot in slots:
        assert slot.expires_at == slot.scheduled_at + timedelta(hours=1)
    for earlier, later in zip(slots, slots[1:]):
        assert earlier.expires_at == later.scheduled_at


# ---------------------------------------------------------------------------
# Item 2 - the scheduler service
# ---------------------------------------------------------------------------


@pytest.fixture()
def mentor_on(monkeypatch):
    """The Settings checkbox, persisted, through the real UiState."""
    from ui.state import UiState

    assert UiState().trade_mentor_enabled is False, "the checkbox defaults OFF"
    state = UiState.load()
    state.trade_mentor_enabled = True
    state.save()
    yield state
    state.trade_mentor_enabled = False
    state.save()


class _Clock:
    """An injected clock. Nothing in these tests sleeps."""

    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment

    def set(self, moment: datetime) -> "_Clock":
        self.moment = moment
        return self


def _service(tmp_path, clock, *, idle=30.0, locked=False, name="slots.json"):
    from ui.services.trade_mentor_service import TradeMentorService

    service = TradeMentorService(
        clock=clock,
        idle_seconds=lambda: idle,
        session_locked=lambda: locked,
        state_path=tmp_path / name,
    )
    due: list = []
    expired: list[str] = []
    service.promptDue.connect(due.append)
    service.promptExpired.connect(expired.append)
    return service, due, expired


def _set_auto_mode(profile: str) -> None:
    """Write the real Auto Pilot state file `read_auto_pilot_mode` reads."""
    from project_paths import AUTOPILOT_STATE_FILE

    path = Path(AUTOPILOT_STATE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"enabled": True, "profile": profile}), encoding="utf-8"
    )


def _clear_auto_mode() -> None:
    from project_paths import AUTOPILOT_STATE_FILE

    Path(AUTOPILOT_STATE_FILE).unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _auto_mode_off():
    """Auto OFF unless a test says otherwise - Mentor does not depend on it."""
    _clear_auto_mode()
    yield
    _clear_auto_mode()


def test_the_slots_file_is_a_named_shared_home_store():
    """`project_paths` owns the path; a service that invents its own would write
    somewhere the rest of the desk cannot find."""
    import project_paths

    assert project_paths.TRADE_MENTOR_SLOTS_FILE.name == "trade_mentor_slots.json"
    assert project_paths.TRADE_MENTOR_SLOTS_FILE.parent == project_paths.PERSISTENT_DATA_DIR
    assert project_paths.TRADE_MENTOR_DRAFTS_FILE.name == "trade_mentor_drafts.json"
    assert project_paths.TRADE_MENTOR_DRAFTS_FILE.parent == project_paths.PERSISTENT_DATA_DIR


def test_a_present_trader_is_asked_once_for_the_hour(tmp_path, mentor_on):
    clock = _Clock(_pacific(NORMAL_SESSION, 9, 0, 30))
    service, due, _expired = _service(tmp_path, clock)

    service.poll()
    clock.set(_pacific(NORMAL_SESSION, 9, 5, 0))
    service.poll()
    clock.set(_pacific(NORMAL_SESSION, 9, 40, 0))
    service.poll()

    assert [slot.slot_id for slot in due] == ["2026-09-14-0900-m5"]
    state = service.slot_state("2026-09-14-0900-m5")
    assert state["delivered_at"] == _pacific(NORMAL_SESSION, 9, 0, 30).isoformat()
    assert not state["answered_at"]
    assert not state["skipped_reason"]


def test_the_mentor_checkbox_and_the_scanner_auto_setting_are_independent(tmp_path, mentor_on):
    """Auto OFF (no state file at all) still prompts; Auto DESK with the Mentor
    checkbox OFF prompts nothing. Neither switch reaches the other."""
    from ui.state import UiState

    clock = _Clock(_pacific(NORMAL_SESSION, 9, 0, 30))
    service, due, _ = _service(tmp_path, clock, name="auto-off.json")
    service.poll()
    assert len(due) == 1, "Auto OFF must not silence the Mentor"

    _set_auto_mode("DESK")
    state = UiState.load()
    state.trade_mentor_enabled = False
    state.save()
    clock2 = _Clock(_pacific(NORMAL_SESSION, 10, 0, 30))
    service2, due2, _ = _service(tmp_path, clock2, name="mentor-off.json")
    service2.poll()
    assert due2 == [], "the Mentor checkbox OFF means no prompt, whatever Auto says"


def test_away_paused_locked_and_idle_each_skip_with_their_own_reason(tmp_path, mentor_on):
    """Four different absences, four different recorded reasons. A single boolean
    "present" would make the coverage gap unreadable later."""
    nine = _pacific(NORMAL_SESSION, 9, 0, 30)
    slot_id = "2026-09-14-0900-m5"

    _set_auto_mode("AWAY")
    away, away_due, _ = _service(tmp_path, _Clock(nine), name="away.json")
    away.poll()
    assert away_due == []
    assert away.slot_state(slot_id)["skipped_reason"] == "away"
    assert not away.slot_state(slot_id)["delivered_at"]
    _clear_auto_mode()

    paused, paused_due, _ = _service(tmp_path, _Clock(nine), name="paused.json")
    paused.pause_today()
    paused.poll()
    assert paused_due == []
    assert paused.slot_state(slot_id)["skipped_reason"] == "paused"

    locked, locked_due, _ = _service(tmp_path, _Clock(nine), locked=True, name="locked.json")
    locked.poll()
    assert locked_due == []
    assert locked.slot_state(slot_id)["skipped_reason"] == "locked"

    from ui.services.trade_mentor_service import IDLE_GRACE_MINUTES

    assert IDLE_GRACE_MINUTES == 20
    idle_secs = IDLE_GRACE_MINUTES * 60 + 1
    idle, idle_due, _ = _service(tmp_path, _Clock(nine), idle=idle_secs, name="idle.json")
    idle.poll()
    assert idle_due == []
    assert idle.slot_state(slot_id)["skipped_reason"] == "idle"


def test_a_trader_quietly_watching_charts_is_present(tmp_path, mentor_on):
    """Idle just inside the grace is NOT away. The trader's brief says so in as many
    words, and an off-by-one on the comparison turns a chart-watching hour into a
    skipped one."""
    from ui.services.trade_mentor_service import IDLE_GRACE_MINUTES

    clock = _Clock(_pacific(NORMAL_SESSION, 9, 0, 30))
    idle_secs = IDLE_GRACE_MINUTES * 60 - 1
    service, due, _ = _service(tmp_path, clock, idle=idle_secs, name="watching.json")
    service.poll()

    assert [slot.slot_id for slot in due] == ["2026-09-14-0900-m5"]


def test_pause_today_silences_today_and_nothing_else(tmp_path, mentor_on):
    """"Pause today" is a day, not a switch. Tuesday's prompts are unaffected."""
    clock = _Clock(_pacific(NORMAL_SESSION, 9, 0, 30))
    service, due, _ = _service(tmp_path, clock)
    service.pause_today()
    service.poll()
    assert due == []

    tuesday = date(2026, 9, 15)
    clock.set(_pacific(tuesday, 9, 0, 30))
    service.poll()
    assert [slot.slot_id for slot in due] == ["2026-09-15-0900-m5"]


def test_a_missed_hour_is_recorded_and_the_next_hour_arrives_alone(tmp_path, mentor_on):
    """No queue, no catch-up burst. The 09:00 slot the trader was away for stays
    skipped forever; 10:00 is delivered on its own."""
    clock = _Clock(_pacific(NORMAL_SESSION, 9, 0, 30))
    service, due, _ = _service(tmp_path, clock)

    _set_auto_mode("AWAY")
    service.poll()
    _clear_auto_mode()

    clock.set(_pacific(NORMAL_SESSION, 10, 0, 30))
    service.poll()

    assert [slot.slot_id for slot in due] == ["2026-09-14-1000-m5_trades"]
    assert service.slot_state("2026-09-14-0900-m5")["skipped_reason"] == "away"
    assert not service.slot_state("2026-09-14-0900-m5")["delivered_at"]


def test_an_unanswered_slot_expires_when_its_hour_runs_out(tmp_path, mentor_on):
    clock = _Clock(_pacific(NORMAL_SESSION, 9, 0, 30))
    service, due, expired = _service(tmp_path, clock)
    service.poll()

    clock.set(_pacific(NORMAL_SESSION, 10, 0, 30))
    service.poll()
    clock.set(_pacific(NORMAL_SESSION, 10, 5, 0))
    service.poll()

    assert expired == ["2026-09-14-0900-m5"], "expired once, not once per tick"
    assert service.slot_state("2026-09-14-0900-m5")["skipped_reason"] == "expired"
    assert [slot.slot_id for slot in due] == [
        "2026-09-14-0900-m5",
        "2026-09-14-1000-m5_trades",
    ]


def test_a_restart_mid_slot_reshows_the_card_without_writing_a_second_slot(tmp_path, mentor_on):
    """A drifted timer, a clock correction or a desk restart must not turn one hour
    into two records. The card comes back because the hour is still open; the slot's
    delivery time is the ORIGINAL one."""
    path = tmp_path / "restart.json"
    clock = _Clock(_pacific(NORMAL_SESSION, 9, 0, 30))
    first, first_due, _ = _service(tmp_path, clock, name="restart.json")
    first.poll()
    assert len(first_due) == 1

    clock2 = _Clock(_pacific(NORMAL_SESSION, 9, 20, 0))
    second, second_due, _ = _service(tmp_path, clock2, name="restart.json")
    second.poll()

    assert [slot.slot_id for slot in second_due] == ["2026-09-14-0900-m5"]
    assert second.slot_state("2026-09-14-0900-m5")["delivered_at"] == _pacific(
        NORMAL_SESSION, 9, 0, 30
    ).isoformat()
    assert path.read_text(encoding="utf-8").count("2026-09-14-0900-m5") == 1


def test_an_answered_slot_is_never_reshown(tmp_path, mentor_on):
    clock = _Clock(_pacific(NORMAL_SESSION, 9, 0, 30))
    service, due, _ = _service(tmp_path, clock)
    service.poll()
    service.mark_answered("2026-09-14-0900-m5")

    clock.set(_pacific(NORMAL_SESSION, 9, 30, 0))
    service.poll()

    assert len(due) == 1
    assert service.slot_state("2026-09-14-0900-m5")["answered_at"]


def test_nothing_is_due_on_a_holiday(tmp_path, mentor_on):
    clock = _Clock(_pacific(HOLIDAY, 9, 0, 30))
    service, due, expired = _service(tmp_path, clock)
    service.poll()

    assert due == []
    assert expired == []
    assert service.next_prompt_at(clock()) is None


def test_settings_can_say_when_the_next_prompt_is(tmp_path, mentor_on):
    clock = _Clock(_pacific(NORMAL_SESSION, 9, 30, 0))
    service, _due, _ = _service(tmp_path, clock)

    assert service.next_prompt_at(clock()) == _pacific(NORMAL_SESSION, 10, 0, 0)


# ---------------------------------------------------------------------------
# Item 3 - the card, and the raw text that reaches the Market Journal
# ---------------------------------------------------------------------------


def _journal(tmp_path):
    """A real MarketJournalService writing a real ledger inside tmp_path."""
    import market_journal
    from evidence_ledger import EvidenceLedger
    from ui.services.market_journal_service import MarketJournalService

    service = MarketJournalService()
    service._ledger = EvidenceLedger(
        stream=market_journal.STREAM,
        schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
        directory=tmp_path / "ledger",
    )
    return service


def _card(tmp_path, journal, clock):
    from ui.widgets.trade_mentor_card import TradeMentorCard

    return TradeMentorCard(
        journal=journal, clock=clock, drafts_path=tmp_path / "drafts.json"
    )


def _nine_slot():
    import trade_mentor_schedule as schedule

    for slot in schedule.slots_for_session(NORMAL_SESSION):
        if slot.scheduled_at.hour == 9:
            return slot
    raise AssertionError("no 09:00 slot")


def test_submit_files_one_market_journal_row_stamped_with_the_real_response_time(tmp_path):
    """A reply typed at 09:12 cannot claim to describe the market at 09:00. The row
    carries BOTH: the slot it answers and the moment it was actually written."""
    journal = _journal(tmp_path)
    clock = _Clock(_pacific(NORMAL_SESSION, 9, 12, 41))
    card = _card(tmp_path, journal, clock)
    slot = _nine_slot()

    card.show_slot(slot)
    card.text_box.setPlainText("SPY lost the 9:45 low; I expect a retest of VWAP.")
    card.submit()

    rows = [
        row
        for row in journal.entries_for("2026-09-14")
        if row.get("origin") == "trade_mentor"
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["text"] == "SPY lost the 9:45 low; I expect a retest of VWAP."
    assert row["timeframe"] == "M5"
    assert row["mentor"]["slot_id"] == "2026-09-14-0900-m5"
    assert row["mentor"]["prompt_kind"] == "m5"
    assert row["mentor"]["scheduled_at"] == slot.scheduled_at.isoformat()
    assert row["mentor"]["responded_at"] == _pacific(NORMAL_SESSION, 9, 12, 41).isoformat()
    # `created_at` is the ledger's own UTC stamp of the same moment, never the
    # scheduled hour - a backdated read is the one thing this row may not be.
    assert row["created_at"].startswith("2026-09-14T16:12:41")


def test_submitting_twice_writes_one_row(tmp_path):
    journal = _journal(tmp_path)
    card = _card(tmp_path, journal, _Clock(_pacific(NORMAL_SESSION, 9, 12, 0)))

    card.show_slot(_nine_slot())
    card.text_box.setPlainText("Chop between the overnight levels.")
    card.submit()
    card.submit()

    rows = [
        row
        for row in journal.entries_for("2026-09-14")
        if row.get("origin") == "trade_mentor"
    ]
    assert len(rows) == 1


def test_read_unchanged_writes_a_new_row_that_references_the_previous_read(tmp_path):
    """A reaffirmation is an OBSERVATION at the current time, not a copy and not a
    correction: the earlier read must still be readable beside it. Writing it through
    `supersedes` would hide the original, which is why that is asserted empty."""
    import trade_mentor_schedule as schedule

    journal = _journal(tmp_path)
    clock = _Clock(_pacific(NORMAL_SESSION, 9, 12, 0))
    card = _card(tmp_path, journal, clock)

    card.show_slot(_nine_slot())
    card.text_box.setPlainText("Bid under the open; buyers in control.")
    card.submit()
    first_id = journal.entries_for("2026-09-14")[-1]["entry_id"]

    eleven = [s for s in schedule.slots_for_session(NORMAL_SESSION) if s.scheduled_at.hour == 11][0]
    clock.set(_pacific(NORMAL_SESSION, 11, 3, 0))
    card.show_slot(eleven, previous=journal.entries_for("2026-09-14")[-1])
    card.read_unchanged()

    rows = [
        row
        for row in journal.entries_for("2026-09-14")
        if row.get("origin") == "trade_mentor"
    ]
    assert len(rows) == 2, "the first read is still current; nothing was superseded"
    reaffirmation = rows[-1]
    assert reaffirmation["reaffirms"] == first_id
    assert not reaffirmation["supersedes"]
    assert reaffirmation["entry_id"] != first_id
    assert reaffirmation["mentor"]["responded_at"] == _pacific(
        NORMAL_SESSION, 11, 3, 0
    ).isoformat()
    assert reaffirmation["text"].strip(), "a reaffirmation still says something"


def test_a_typed_draft_survives_the_next_hour_and_is_never_a_read(tmp_path):
    """The new hour replaces the card. Whatever was half-typed is kept where the
    trader can get it back - and is NOT in the journal, because they never pressed
    Submit and an unanswered prompt is no observation."""
    journal = _journal(tmp_path)
    clock = _Clock(_pacific(NORMAL_SESSION, 9, 20, 0))
    card = _card(tmp_path, journal, clock)
    nine = _nine_slot()

    card.show_slot(nine)
    card.text_box.setPlainText("half a thought about the ")

    import trade_mentor_schedule as schedule

    ten = [s for s in schedule.slots_for_session(NORMAL_SESSION) if s.scheduled_at.hour == 10][0]
    clock.set(_pacific(NORMAL_SESSION, 10, 0, 30))
    card.show_slot(ten)

    assert card.text_box.toPlainText() == "", "the new hour starts clean"
    assert card.draft_for(nine.slot_id) == "half a thought about the "
    saved = json.loads((tmp_path / "drafts.json").read_text(encoding="utf-8"))
    assert "half a thought about the " in json.dumps(saved)
    assert journal.entries_for("2026-09-14") == []


def test_skip_records_the_traders_own_reason_and_writes_no_read(tmp_path):
    journal = _journal(tmp_path)
    card = _card(tmp_path, journal, _Clock(_pacific(NORMAL_SESSION, 9, 12, 0)))
    slot = _nine_slot()

    card.show_slot(slot)
    card.text_box.setPlainText("not now")
    result = card.skip()

    assert result["skipped_reason"] == "trader_skip"
    assert result["slot_id"] == slot.slot_id
    assert journal.entries_for("2026-09-14") == []


def test_give_a_read_is_available_with_no_slot_due(tmp_path):
    """The manual door. It is a real read at a real time, not a scheduled one."""
    journal = _journal(tmp_path)
    clock = _Clock(_pacific(NORMAL_SESSION, 14, 40, 0))
    card = _card(tmp_path, journal, clock)

    card.give_a_read()
    card.text_box.setPlainText("After the bell: closed on the highs.")
    card.submit()

    rows = [
        row
        for row in journal.entries_for("2026-09-14")
        if row.get("origin") == "trade_mentor"
    ]
    assert len(rows) == 1
    assert rows[0]["mentor"]["prompt_kind"] == "manual"
    assert rows[0]["mentor"]["responded_at"] == _pacific(
        NORMAL_SESSION, 14, 40, 0
    ).isoformat()


def test_the_card_never_takes_focus_from_what_the_trader_is_doing(tmp_path):
    """Non-modal, no focus stealing. The trader may be typing a symbol into the chart
    box when the hour turns; a `setFocus()` on the card's text area would eat the next
    keystroke - and offscreen Qt reports that theft faithfully."""
    journal = _journal(tmp_path)
    card = _card(tmp_path, journal, _Clock(_pacific(NORMAL_SESSION, 9, 0, 30)))

    host = QWidget()
    layout = QVBoxLayout(host)
    typing_here = QLineEdit(host)
    layout.addWidget(typing_here)
    card.setParent(host)
    layout.addWidget(card)
    host.show()
    host.activateWindow()
    typing_here.setFocus()
    _app.processEvents()

    active_before = QApplication.activeWindow()
    card.show_slot(_nine_slot())
    _app.processEvents()

    assert QApplication.focusWidget() is typing_here
    assert QApplication.activeWindow() is active_before
    assert not card.isModal()
    host.close()


def test_the_chart_host_docks_the_card_and_keeps_a_give_a_read_button(tmp_path):
    """Under the chart, beside the arm bar - and the arm bar stays exactly where it
    is. Hidden until something is due, with the manual button always reachable."""
    from ui.widgets.alert_chart_review import AlertChartReview
    from ui.widgets.trade_mentor_card import TradeMentorCard

    review = AlertChartReview(dock_arm_bar=True)
    review.show()
    _app.processEvents()

    cards = review.findChildren(TradeMentorCard)
    assert len(cards) == 1, "the host owns exactly one mentor card"
    assert not cards[0].isVisible(), "nothing is due, so nothing is shown"

    buttons = [
        button
        for button in review.findChildren(QPushButton)
        if "give a read" in button.text().strip().lower()
    ]
    assert len(buttons) == 1
    assert buttons[0].isEnabled()
    review.close()


# ---------------------------------------------------------------------------
# Item 4 - the 10:00 card's second section
# ---------------------------------------------------------------------------


def _store(tmp_path):
    from journal_store import JournalStore

    return JournalStore(tmp_path / "journal.sqlite3")


def _seed_trade(store, trade_id, symbol, trade_date="2026-09-11", status="CLOSED"):
    with store.connection() as conn:
        conn.execute(
            """
            INSERT INTO trades(
                trade_id, broker, account_number, symbol, direction, status,
                opened_at, closed_at, trade_date, updated_at
            ) VALUES(?, 'QUESTRADE', '123', ?, 'LONG', ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                symbol,
                status,
                f"{trade_date}T06:45:00-07:00",
                f"{trade_date}T09:10:00-07:00" if status == "CLOSED" else None,
                trade_date,
                "2026-09-11T00:00:00",
            ),
        )
    return trade_id


def _cover(store, day="2026-09-11"):
    import journal_coverage

    journal_coverage.mark_coverage(
        store,
        broker="QUESTRADE",
        account_number="123",
        day=day,
        status=journal_coverage.COVERED,
        source="test",
    )


def test_monday_asks_about_friday(tmp_path):
    """The PREVIOUS EXCHANGE SESSION, walked on the calendar - not "yesterday"."""
    import trade_mentor_trade_check as check

    store = _store(tmp_path)
    _cover(store)
    _seed_trade(store, "T1", "AAPL")

    task = check.build_task(store, NORMAL_SESSION)

    assert task.reviewed_session == "2026-09-11"
    assert [question.trade_id for question in task.trades] == ["T1"]


def test_only_the_fields_that_are_actually_missing_are_asked_for(tmp_path):
    """"All support data" is not an ever-growing compulsory questionnaire. A trade
    that already carries its thesis, its setup claim and its stop is asked one
    question, not four."""
    import trade_mentor_trade_check as check

    store = _store(tmp_path)
    _cover(store)
    _seed_trade(store, "T1", "AAPL")
    _seed_trade(store, "T2", "MSFT")
    store.save_trade_annotation("T2", setup_tags="ORB", notes="broke the opening range")
    store.save_risk_fields("T2", planned_stop=181.40)

    task = check.build_task(store, NORMAL_SESSION)
    asked = {question.trade_id: tuple(question.missing) for question in task.trades}

    assert set(check.MATERIAL_FIELDS) == {"thesis", "stop", "target", "setup"}
    assert asked["T1"] == check.MATERIAL_FIELDS
    assert asked["T2"] == ("target",)


def test_the_four_answer_states_stay_distinct_in_the_saved_row(tmp_path):
    """not supplied / no fixed target / not remembered / not applicable are four
    different facts. Collapsing any two of them destroys the only thing this task
    exists to record."""
    import trade_mentor_trade_check as check

    store = _store(tmp_path)
    _cover(store)
    _seed_trade(store, "T1", "AAPL")
    written_at = _pacific(NORMAL_SESSION, 10, 6, 12)

    check.save_answers(
        store,
        "T1",
        {
            "thesis": {"state": check.ANSWER_NOT_REMEMBERED},
            "target": {"state": check.ANSWER_NO_FIXED_TARGET},
            "stop": {"state": check.ANSWER_NOT_APPLICABLE, "text": "exit if the H1 level fails"},
            "setup": {"state": check.ANSWER_NOT_SUPPLIED},
        },
        now=written_at,
    )

    assert len({
        check.ANSWER_NOT_SUPPLIED,
        check.ANSWER_NO_FIXED_TARGET,
        check.ANSWER_NOT_REMEMBERED,
        check.ANSWER_NOT_APPLICABLE,
    }) == 4
    rows = {row["field"]: row for row in check.recalled_fields(store, "T1")}
    assert rows["thesis"]["state"] == check.ANSWER_NOT_REMEMBERED
    assert rows["target"]["state"] == check.ANSWER_NO_FIXED_TARGET
    assert rows["stop"]["state"] == check.ANSWER_NOT_APPLICABLE
    assert rows["setup"]["state"] == check.ANSWER_NOT_SUPPLIED
    assert rows["stop"]["text"] == "exit if the H1 level fails"


def test_a_next_day_answer_says_when_it_was_written_and_that_it_is_recalled(tmp_path):
    """Remembered risk is never presented as a documented pre-entry plan."""
    import trade_mentor_trade_check as check

    store = _store(tmp_path)
    _cover(store)
    _seed_trade(store, "T1", "AAPL")
    written_at = _pacific(NORMAL_SESSION, 10, 6, 12)

    check.save_answers(
        store,
        "T1",
        {"thesis": {"state": check.ANSWER_NOT_SUPPLIED, "text": "gap fill into the 50dma"}},
        now=written_at,
    )

    row = check.recalled_fields(store, "T1")[0]
    assert row["recalled_after_session"] is True
    assert row["recorded_at"] == written_at.isoformat()
    assert row["trade_date"] == "2026-09-11"
    assert row["recorded_at"][:10] == "2026-09-14", "written the NEXT session, and it says so"


def test_no_fixed_target_is_a_complete_answer_and_no_stop_is_never_a_zero(tmp_path):
    """Two separate refusals. "I had no target" answers the question and must stop
    being asked; "I had no stop" must never become `planned_stop = 0.0`, which would
    read downstream as a stop at zero and an infinite R."""
    import trade_mentor_trade_check as check

    store = _store(tmp_path)
    _cover(store)
    _seed_trade(store, "T1", "AAPL")
    store.save_trade_annotation("T1", setup_tags="ORB", notes="broke the opening range")

    check.save_answers(
        store,
        "T1",
        {
            "target": {"state": check.ANSWER_NO_FIXED_TARGET},
            "stop": {"state": check.ANSWER_NOT_APPLICABLE},
        },
        now=_pacific(NORMAL_SESSION, 10, 6, 12),
    )

    task = check.build_task(store, NORMAL_SESSION)
    assert task.trades == (), "every material field is now answered"

    rows = {row["field"]: row for row in check.recalled_fields(store, "T1")}
    assert rows["stop"]["value"] is None
    assert rows["target"]["value"] is None
    with store.connection() as conn:
        planned_stop = conn.execute(
            "SELECT planned_stop FROM trade_annotations WHERE trade_id = 'T1'"
        ).fetchone()[0]
    assert planned_stop is None, "a missing stop is not a stop at zero"


def test_the_morning_task_is_capped_and_the_rest_is_counted_not_forgotten(tmp_path):
    """Five minutes or three trades. The remainder is a NUMBER the Journal's
    completeness view shows, never a fourth question and never silence."""
    import trade_mentor_trade_check as check

    store = _store(tmp_path)
    _cover(store)
    for index in range(5):
        _seed_trade(store, f"T{index}", f"SYM{index}")

    task = check.build_task(store, NORMAL_SESSION)

    assert check.TRADE_CAP_DEFAULT == 3
    assert len(task.trades) == 3
    assert task.remaining == 2
    assert task.incomplete_total == 5


def test_missing_broker_coverage_says_journal_not_ready_rather_than_no_trades(tmp_path):
    """An empty questionnaire because the statement has not landed is a lie about the
    session. It says so instead, and asks nothing."""
    import trade_mentor_trade_check as check

    store = _store(tmp_path)
    _seed_trade(store, "T1", "AAPL")

    task = check.build_task(store, NORMAL_SESSION)

    assert task.journal_ready is False
    assert task.trades == ()
    assert task.reason == "journal not ready"


def test_a_covered_session_with_complete_trades_asks_nothing(tmp_path):
    import trade_mentor_trade_check as check

    store = _store(tmp_path)
    _cover(store)

    task = check.build_task(store, NORMAL_SESSION)

    assert task.journal_ready is True
    assert task.trades == ()
    assert task.remaining == 0
    assert task.reason == ""


# ---------------------------------------------------------------------------
# Item 5 - Settings
# ---------------------------------------------------------------------------


def test_settings_carries_the_checkbox_the_pause_and_the_timezone_sentence():
    """The trader has to be told which clock this runs on, in the place they turn it
    on. "PST" is a wall clock with daylight saving, and the panel says so."""
    from ui.panels.settings_panel import SettingsPanel
    from ui.state import UiState

    state = UiState.load()
    state.trade_mentor_enabled = False
    state.save()
    panel = SettingsPanel(state)

    assert isinstance(panel.trade_mentor_input, QCheckBox)
    assert panel.trade_mentor_input.isChecked() is False
    assert "Trade Mentor" in panel.trade_mentor_input.text()

    panel.trade_mentor_input.setChecked(True)
    _app.processEvents()
    assert panel.state.trade_mentor_enabled is True
    assert UiState.load().trade_mentor_enabled is True, "the checkbox is persisted"

    labels = " ".join(label.text() for label in panel.findChildren(QLabel))
    assert "America/Los_Angeles" in labels
    assert "07:00" in labels

    pause = [
        button
        for button in panel.findChildren(QPushButton)
        if "pause today" in button.text().strip().lower()
    ]
    assert len(pause) == 1

    state.trade_mentor_enabled = False
    state.save()
