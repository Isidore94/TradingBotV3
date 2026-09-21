"""TJ-9 items 1 and 2 - the trade check moves to 09:00 and stops being optional.

Written BEFORE the fix and red on `claude/tj9-forced-trade-labels`'s base commit
(`57151d0f`). The builder makes them pass and may only ADD.

WHAT IS PINNED, AND WHY IT IS A NUMBER
--------------------------------------
* **Item 1** is pinned as the exact `(wall time, kind, post_close)` tuple of a
  whole session, on a normal day, on a real early close and across both DST
  transitions - the same shape `tests/test_ws_tm_trade_mentor.py` pins for the
  10:00 schedule, moved to 09:00. An implementation that merely renames a
  constant without following `_kind_for` produces a different tuple. The early
  close is the load-bearing case: with `TRADES_HOUR = 9` the 10:00 slot on a
  short day is no longer forced into existence at all, so the tuple LOSES a
  slot rather than relabelling one.
* **Item 2** is pinned as a COUNT: a session with four incomplete trades lists
  four, not `TRADE_CAP_DEFAULT`. The cap constant is asserted to still be 3, so
  "delete the cap" is not a way to pass.
* The Save gate is driven through the real widget offscreen: the button's own
  `isEnabled()` after real `QComboBox` selections, not a helper that returns a
  boolean.
* The ride is driven through `show_slot`, which is what actually clears the
  section today (`trade_mentor_card.py:368`).

`TRADE_CAP_DEFAULT` is read by name, never copied as a literal, and no test
here asserts a vocabulary version.
"""

from __future__ import annotations

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
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj9_support import (  # noqa: E402
    REVIEWED,
    SESSION_TODAY,
    add_round_trip,
    mark_covered,
    new_store,
    slot_at,
)

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Trade Mentor card is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

#: The day after Thanksgiving 2026 - a real early close at 13:00 ET = 10:00 PT.
EARLY_CLOSE_SESSION = date(2026, 11, 27)
PACIFIC = ZoneInfo("America/Los_Angeles")


def _shape(slots) -> tuple[tuple[str, str, bool], ...]:
    return tuple(
        (slot.scheduled_at.strftime("%H:%M"), slot.kind, bool(slot.post_close))
        for slot in slots
    )


# ---------------------------------------------------------------------------
# Item 1 - 09:00
# ---------------------------------------------------------------------------


def test_the_previous_sessions_trades_are_asked_at_nine_and_ten_is_a_plain_m5_read():
    """The trader asked to be forced to label "around 0900". The trade check is
    the 09:00 slot; 10:00 goes back to being an ordinary tape read."""
    import trade_mentor_schedule as schedule

    assert schedule.TRADES_HOUR == 9
    assert _shape(schedule.slots_for_session(SESSION_TODAY)) == (
        ("07:00", "m5", False),
        ("08:00", "m5_d1", False),
        ("09:00", "m5_trades", False),
        ("10:00", "m5", False),
        ("11:00", "m5", False),
        ("12:00", "m5_d1", False),
    )
    ids = [slot.slot_id for slot in schedule.slots_for_session(SESSION_TODAY)]
    assert "2026-09-14-0900-m5_trades" in ids
    assert "2026-09-14-1000-m5" in ids
    # The collision rule is unchanged: one card at one instant.
    assert len(ids) == len(set(ids))
    assert schedule.FIRST_HOUR == 7 and schedule.D1_HOURS == (8, 12)


def test_an_early_close_asks_at_nine_and_no_longer_forces_a_ten_oclock_slot():
    """2026-11-27 closes at 10:00 Pacific. With the check at 09:00 it is inside
    the hourly window and is NOT post-close, and nothing forces a 10:00 slot into
    existence any more - so the short day loses a slot rather than relabelling
    one. The noon D1 read still survives, labelled post-close."""
    import trade_mentor_schedule as schedule

    assert _shape(schedule.slots_for_session(EARLY_CLOSE_SESSION)) == (
        ("07:00", "m5", False),
        ("08:00", "m5_d1", False),
        ("09:00", "m5_trades", False),
        ("12:00", "m5_d1", True),
    )
    assert 10 not in {s.scheduled_at.hour for s in schedule.slots_for_session(EARLY_CLOSE_SESSION)}


def test_the_nine_oclock_check_follows_the_pacific_wall_clock_across_a_dst_change():
    """09:00 in the trader's kitchen, in March and in November. A fixed UTC-8
    implementation files half the year's checks an hour away from the session
    they are about, and the KIND must travel with the hour, not with the
    offset."""
    import trade_mentor_schedule as schedule

    def check_slot(day: date):
        for slot in schedule.slots_for_session(day):
            if slot.kind == schedule.KIND_M5_TRADES:
                return slot
        raise AssertionError(f"no trade-check slot on {day}")

    before_spring = check_slot(date(2026, 3, 6))
    after_spring = check_slot(date(2026, 3, 9))
    before_fall = check_slot(date(2026, 10, 30))
    after_fall = check_slot(date(2026, 11, 2))

    assert [s.scheduled_at.hour for s in (before_spring, after_spring, before_fall, after_fall)] == [
        9,
        9,
        9,
        9,
    ]
    assert before_spring.scheduled_at.utcoffset() == timedelta(hours=-8)
    assert after_spring.scheduled_at.utcoffset() == timedelta(hours=-7)
    assert before_fall.scheduled_at.utcoffset() == timedelta(hours=-7)
    assert after_fall.scheduled_at.utcoffset() == timedelta(hours=-8)
    assert str(after_spring.scheduled_at.tzinfo) == "America/Los_Angeles"


# ---------------------------------------------------------------------------
# Item 2 - forced
# ---------------------------------------------------------------------------


def test_every_trade_of_the_previous_session_is_listed_even_past_the_cap(tmp_path):
    """Four trades on Friday means four sections on Monday's card. The cap of
    three was five minutes of questions; the trader asked to be FORCED, so it no
    longer applies to the session being reviewed. The constant itself stays -
    deleting it is not a way to pass, because the older backlog still uses it."""
    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    for symbol in ("AAPL", "MSFT", "NVDA", "TSLA"):
        add_round_trip(store, symbol)

    task = check.build_task(store, SESSION_TODAY)

    assert check.TRADE_CAP_DEFAULT == 3
    assert task.reviewed_session == REVIEWED
    assert task.journal_ready is True
    assert len(task.trades) == 4
    assert {q.symbol for q in task.trades} == {"AAPL", "MSFT", "NVDA", "TSLA"}
    assert task.remaining == 0
    assert task.incomplete_total == 4


def test_save_is_disabled_until_every_listed_field_holds_a_value_or_an_answer_state(tmp_path):
    """Forced means the button is grey. One open field keeps it grey; an
    explicit `not_remembered` - which is a complete answer, not a blank - turns
    it on. The four answer states are read from the module, never spelled here."""
    import trade_mentor_trade_check as check
    from ui.widgets.trade_mentor_card import TradeMentorCard

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    trade_id = add_round_trip(store, "AAPL")
    # The setup is already the trader's own, so only thesis/stop/target are asked
    # and this test stays clear of item 3's confirm button.
    store.save_trade_annotation(trade_id, setup_tags="vwap_reclaim", notes="")

    task = check.build_task(store, SESSION_TODAY)
    assert len(task.trades) == 1
    asked = task.trades[0].missing
    assert set(asked) == {"thesis", "stop", "target"}

    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    card.set_trade_check(task, store=store)

    assert card.save_answers_button.isVisibleTo(card) is True
    assert card.save_answers_button.isEnabled() is False, "nothing is answered yet"

    combos = {name: card._answer_inputs[trade_id][name][0] for name in asked}
    combos["thesis"].setCurrentIndex(combos["thesis"].findData(check.ANSWER_NOT_SUPPLIED))
    combos["stop"].setCurrentIndex(combos["stop"].findData(check.ANSWER_NOT_APPLICABLE))
    assert card.save_answers_button.isEnabled() is False, "target is still open"

    combos["target"].setCurrentIndex(combos["target"].findData(check.ANSWER_NOT_REMEMBERED))
    # LEAD AMENDMENT 2026-09-21 (TJ-9E, the trader's own request): a round-trip
    # trade now also carries ONE forced EXIT box - "Why did you exit? What did
    # you feel? What were you watching?" - so Save waits on it as it waits on
    # the entry fields. One click answers it; what this test pins is unchanged.
    assert card.save_answers_button.isEnabled() is False, "the exit box is still open"
    card.set_exit_answer_state(trade_id, check.ANSWER_NOT_REMEMBERED)
    assert card.save_answers_button.isEnabled() is True

    # And going back to the blank "-" closes it again: the gate is a state, not
    # a one-way latch.
    combos["stop"].setCurrentIndex(combos["stop"].findData(""))
    assert card.save_answers_button.isEnabled() is False


def test_an_unanswered_trade_section_rides_to_the_next_slot_of_the_same_session(tmp_path):
    """An unanswered card must not expire into silence. The 10:00 card carries
    the same section, and the next SESSION's first card does not - a question
    about Friday's trades asked on Wednesday is a different question."""
    import trade_mentor_trade_check as check
    from ui.widgets.trade_mentor_card import TradeMentorCard

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    trade_id = add_round_trip(store, "AAPL")

    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    card.show_slot(slot_at(SESSION_TODAY, 9))
    card.set_trade_check(check.build_task(store, SESSION_TODAY), store=store)
    assert card.trade_check_box.isVisibleTo(card) is True

    card.show_slot(slot_at(SESSION_TODAY, 10))
    assert card.trade_check_box.isVisibleTo(card) is True, "the section rides to 10:00"
    assert trade_id in card._answer_inputs
    assert card.save_answers_button.isVisibleTo(card) is True

    card.show_slot(slot_at(SESSION_TODAY, 11))
    assert card.trade_check_box.isVisibleTo(card) is True, "and to 11:00"

    # A new session starts clean.
    card.show_slot(slot_at(date(2026, 9, 15), 7))
    assert card.trade_check_box.isVisibleTo(card) is False
    assert card._answer_inputs == {}


def test_the_service_counts_the_unlabelled_trades_of_a_session(tmp_path):
    """`unlabelled_trades(session)` is the number a later reader prints. It is
    the count of trades ON that session that still cannot answer every material
    field - here two of four, because two were answered in full."""
    import trade_mentor_trade_check as check
    from ui.services.trade_mentor_service import TradeMentorService

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    ids = {symbol: add_round_trip(store, symbol) for symbol in ("AAPL", "MSFT", "NVDA", "TSLA")}
    for symbol in ("AAPL", "MSFT"):
        store.save_trade_annotation(ids[symbol], setup_tags="vwap_reclaim", notes="held the level")
        store.save_risk_fields(ids[symbol], planned_stop=9.5, risk_source="manual")
        check.save_answers(
            store,
            ids[symbol],
            {"target": {"state": check.ANSWER_NO_FIXED_TARGET, "text": ""}},
        )

    service = TradeMentorService(
        clock=lambda: datetime(2026, 9, 14, 9, 5, tzinfo=PACIFIC),
        idle_seconds=lambda: 0.0,
        session_locked=lambda: False,
        state_path=tmp_path / "slots.json",
    )

    assert service.unlabelled_trades(REVIEWED, store=store) == 2
    assert service.unlabelled_trades("2026-09-10", store=store) == 0
