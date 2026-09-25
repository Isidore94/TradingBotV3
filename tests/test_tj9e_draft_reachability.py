r"""TJ-9E - the Confirm click ARRIVES. Review 1's draft-reachability blocker.

The trader asked for half of this feature in one clause: *"ideally we can just
write it out and the AI fills this stuff in overnight"* (2026-09-21). As first
built, the second half could never be collected - the night wrote its reading
and the trader was never shown it. Two reasons, and the second is the one that
makes the first insufficient to fix:

* since review 1 blocker 5 an exit the trader has EXPLAINED is answered, so a
  row with nothing else open leaves the card;
* **the timeline never lines up.** The trade exits MONDAY. The note is typed on
  TUESDAY's 09:00 card, which reviews Monday. TUESDAY NIGHT drafts it. And
  WEDNESDAY's 09:00 card reviews TUESDAY - where Monday's trade is not a row at
  all, however many questions it still owes.

So a waiting reading is offered on its OWN clock:
`check.EXIT_DRAFT_OFFER_SESSIONS` exchange sessions, independent of the session
the card reviews, through the lane the registry already had
(`MainWindow._mentor_exit_drafts` -> `state["exit_drafts"]` ->
`exit_draft_review`). The registry decides WHICH and HOW MANY; the card draws
the row and its own Confirm / Correct are still the one writer of a confirmed
exit row.

NO MODEL RUNS HERE: every night in this file is `fx.fake_request`.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj9e_support as fx  # noqa: E402

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Trade Mentor card is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

#: Monday 2026-09-14 is the exit. Tuesday's card reviews it, Tuesday night
#: drafts it, and WEDNESDAY's card - which reviews Tuesday - is where the
#: trader must be shown it.
MONDAY = "2026-09-14"
TUESDAY = date(2026, 9, 15)
TUESDAY_NIGHT = datetime.fromisoformat("2026-09-16T02:00:00-04:00")
WEDNESDAY = date(2026, 9, 16)
THURSDAY = date(2026, 9, 17)


def _execution(execution_id, *, symbol, side, qty, price, timestamp):
    from journal_importers import manual_execution_from_fields

    return manual_execution_from_fields(
        {
            "broker": "MANUAL",
            "account_number": fx.ACCOUNT,
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "price": price,
            "timestamp": timestamp,
            "security_type": "STK",
            "currency": "USD",
            "commission": 0,
            "fees": 0,
            "execution_id": execution_id,
        }
    )


def _day_trade(store, symbol: str, day: str) -> str:
    """One round trip opened AND closed on `day`, through the real assembler."""
    store.upsert_executions(
        [
            _execution(f"{symbol}-{day}-1", symbol=symbol, side="BUY", qty=100,
                       price=20.0, timestamp=f"{day}T07:35:00"),
            _execution(f"{symbol}-{day}-2", symbol=symbol, side="SELL", qty=100,
                       price=21.0, timestamp=f"{day}T09:05:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    return str(
        [row for row in store.list_trades(trade_date=day)
         if str(row.get("symbol")) == symbol][0]["trade_id"]
    )


def _night(store, session: str, root: Path, *, symbols=1) -> None:
    """One fake night over `session`'s waiting notes. No model is loaded."""
    import exit_reasons
    import trader_state_tags
    from ai_jobs import exit_note_fields

    out = exit_note_fields.run_exit_note_fields(
        session_date=session,
        now=TUESDAY_NIGHT,
        root=root,
        store=store,
        request=fx.fake_request(
            fx.good_reply(exit_reasons.codes()[0], trader_state_tags.codes()[:1])
        ),
    )
    assert out["status"] == "ok", out
    assert out["extra"]["drafted"] == symbols, out


def _monday_exit_drafted(tmp_path, *, symbols=("DAYT",)):
    """The whole timeline: exit Monday, note Tuesday, draft Tuesday night."""
    import trade_mentor_trade_check as check

    store = fx.new_store(tmp_path)
    fx.mark_covered(store, MONDAY)
    ids = {symbol: _day_trade(store, symbol, MONDAY) for symbol in symbols}
    for index, trade_id in enumerate(ids.values()):
        check.save_exit_note(
            store,
            trade_id,
            fx.EXIT_NOTE,
            exit_session=MONDAY,
            # Typed on TUESDAY's 09:00 card, a minute apart so "oldest first"
            # is a real ordering.
            now=datetime.fromisoformat(f"2026-09-15T09:0{index}:00-04:00"),
        )
    root = tmp_path / "packs"
    _night(store, MONDAY, root, symbols=len(ids))
    return store, ids, root


def _state(window, store, session: date, *, auto_mode: str = "DESK"):
    """The lane payload the real `MainWindow` builds for a card."""
    from trade_mentor_schedule import slots_for_session

    slot = [item for item in slots_for_session(session) if item.scheduled_at.hour == 9][0]
    window._auto_mode_now = lambda: auto_mode
    return slot, window._mentor_question_state(slot, store)


@pytest.fixture()
def window(tmp_path, monkeypatch):
    """A real `MainWindow`. Every store it is asked about is handed IN."""
    from ui.app import MainWindow
    from ui.state import UiState

    QApplication.instance() or QApplication([])
    made = MainWindow(UiState(workspace_mode="workspace"))
    try:
        yield made
    finally:
        made.close()


def _point_the_lane_at(monkeypatch, root: Path) -> None:
    """The lane reads the desk's OWN pack folder; point it at this test's."""
    from ai_jobs import exit_note_fields

    real = exit_note_fields.read_latest
    monkeypatch.setattr(
        exit_note_fields,
        "read_latest",
        lambda session, root=None: real(session, root=root if root is not None else _POINTED[0]),
    )
    _POINTED[0] = root


_POINTED: list[Path] = [None]


# ---------------------------------------------------------------------------
# the headline: Wednesday's card, reviewing a session Monday's trade is not in
# ---------------------------------------------------------------------------
def test_wednesdays_card_offers_mondays_reading_although_it_reviews_tuesday(
    tmp_path, monkeypatch, window
):
    """THE BLOCKER. Hand-counted: Tuesday has 0 trades, and the card carries 1
    draft row - the symbol, the trader's own words and both verbs.

    Before this the lane was keyed to the session the card REVIEWED, so this
    card offered nothing and the reading was never seen by anybody.
    """
    import trade_mentor_trade_check as check

    store, ids, root = _monday_exit_drafted(tmp_path)
    _point_the_lane_at(monkeypatch, root)
    trade_id = ids["DAYT"]

    # Wednesday reviews TUESDAY, and Tuesday has no trades at all.
    assert check.previous_exchange_session(WEDNESDAY) == TUESDAY.isoformat()
    assert store.list_trades(trade_date=TUESDAY.isoformat()) == []

    lane = window._mentor_exit_drafts(store, WEDNESDAY.isoformat())
    assert [row["trade_id"] for row in lane] == [trade_id], lane
    assert lane[0]["exit_session"] == MONDAY
    assert lane[0]["raw_text"] == fx.EXIT_NOTE
    key = check.exit_key(trade_id, MONDAY)
    assert lane[0]["key"] == key

    import mentor_questions

    slot, state = _state(window, store, WEDNESDAY)
    result = mentor_questions.pending(state, slot)
    offered = [item for item in result.asked if item.kind == "exit_draft_review"]
    assert len(offered) == 1, [item.kind for item in result.asked]
    # The subject IS the reading, not the trade (review 2 blocker 1).
    assert offered[0].subject_id == key

    card = window.trading_panel.alert_center.chart_review.mentor_card
    card.show_slot(slot)
    card.set_questions(result, store=store, service=window.trade_mentor_service)

    assert "DAYT" in card.question_prompt_text("exit_draft_review", key)
    assert MONDAY in card.question_prompt_text("exit_draft_review", key)
    assert card.exit_draft_line(trade_id).strip(), "no draft line was drawn"
    assert card.exit_confirm_button(trade_id) is not None
    assert card.exit_correct_button(trade_id) is not None
    # It is NOT a combo: a reading is confirmed or corrected, never picked.
    assert card.question_box("exit_draft_review", key) is None


def test_confirm_writes_one_row_and_the_next_card_offers_nothing(
    tmp_path, monkeypatch, window
):
    """Once signed off it is gone for good - the confirmed row is the fact.

    Hand-counted: 1 `EXIT_NOTE_FIELDS` row after the click, 0 drafts offered on
    Thursday, and the row leaves Wednesday's card as soon as the write lands.
    """
    import trade_mentor_trade_check as check

    store, ids, root = _monday_exit_drafted(tmp_path)
    _point_the_lane_at(monkeypatch, root)
    trade_id = ids["DAYT"]

    assert check.exit_fields(store, trade_id) == {}
    check.confirm_exit_fields(
        store,
        trade_id,
        window._mentor_exit_drafts(store, WEDNESDAY.isoformat())[0],
        now=datetime.fromisoformat("2026-09-16T09:05:00-04:00"),
    )

    rows = store.list_opportunity_events(
        trade_id=trade_id, event_type=check.EVENT_EXIT_FIELDS, limit=100
    )
    assert len(rows) == 1, rows
    assert check.exit_fields(store, trade_id)["status"] == "confirmed"
    assert window._mentor_exit_drafts(store, THURSDAY.isoformat()) == []


def test_the_card_drops_the_row_the_moment_the_trader_confirms(
    tmp_path, monkeypatch, window
):
    """The click, through the card's own button and its one-write-in-flight
    worker. Hand-counted: 1 row before, 0 after, and Save untouched."""
    import time

    import mentor_questions
    import trade_mentor_trade_check as check

    store, ids, root = _monday_exit_drafted(tmp_path)
    _point_the_lane_at(monkeypatch, root)
    trade_id = ids["DAYT"]

    slot, state = _state(window, store, WEDNESDAY)
    card = window.trading_panel.alert_center.chart_review.mentor_card
    card.show_slot(slot)
    card.set_questions(
        mentor_questions.pending(state, slot),
        store=store,
        service=window.trade_mentor_service,
    )
    assert card.exit_draft_line(trade_id).strip()

    card.exit_confirm_button(trade_id).click()
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and card.exit_write_in_flight(trade_id):
        _app.processEvents()
    assert card.exit_write_in_flight(trade_id) is False, "the card never settled"
    assert "not" not in card.status_label.text().lower(), card.status_label.text()
    assert card.exit_draft_line(trade_id) == "", "the signed-off row stayed"
    assert card.question_box("exit_draft_review", check.exit_key(trade_id, MONDAY)) is None


# ---------------------------------------------------------------------------
# review 2 blocker 1 - a reading belongs to (trade, EXIT SESSION)
# ---------------------------------------------------------------------------
def _two_session_exit_drafted(tmp_path):
    """ONE trade, half closed Monday and half closed Tuesday, read twice.

    The live shape of 13 of the journal's 216 trades, and the one the packet's
    own correction 6 says is "drafted separately, confirmed separately".
    """
    import trade_mentor_trade_check as check

    store = fx.new_store(tmp_path)
    fx.mark_covered(store, MONDAY)
    fx.mark_covered(store, TUESDAY.isoformat())
    store.upsert_executions(
        [
            _execution("TWO-1", symbol="SCLO", side="BUY", qty=100, price=40.0,
                       timestamp=f"{MONDAY}T07:31:00"),
            _execution("TWO-2", symbol="SCLO", side="SELL", qty=40, price=41.0,
                       timestamp=f"{MONDAY}T10:05:00"),
            _execution("TWO-3", symbol="SCLO", side="SELL", qty=60, price=43.0,
                       timestamp=f"{TUESDAY.isoformat()}T10:05:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    trade_id = str(store.list_trades(trade_date=TUESDAY.isoformat())[0]["trade_id"])
    assert check.exit_sessions(store, trade_id) == (MONDAY, TUESDAY.isoformat())

    root = tmp_path / "packs"
    for index, session in enumerate((MONDAY, TUESDAY.isoformat())):
        check.save_exit_note(
            store,
            trade_id,
            f"{fx.EXIT_NOTE} ({session})",
            exit_session=session,
            now=datetime.fromisoformat(f"2026-09-15T09:0{index}:00-04:00"),
        )
        _night(store, session, root)
    return store, trade_id, root


def test_a_trade_read_twice_is_offered_twice_one_row_per_exit_session(
    tmp_path, monkeypatch, window
):
    """Review 2 blocker 1. Hand-counted: 2 exit sessions, 2 readings, 2 rows.

    Keyed by the trade alone the second reading was never offered, never
    carried and never said - it simply stayed `provisional` for good.
    """
    import mentor_questions
    import trade_mentor_trade_check as check

    store, trade_id, root = _two_session_exit_drafted(tmp_path)
    _point_the_lane_at(monkeypatch, root)

    lane = window._mentor_exit_drafts(store, WEDNESDAY.isoformat())
    assert [row["exit_session"] for row in lane] == [MONDAY, TUESDAY.isoformat()], lane
    assert [row["key"] for row in lane] == [
        check.exit_key(trade_id, MONDAY),
        check.exit_key(trade_id, TUESDAY.isoformat()),
    ]

    slot, state = _state(window, store, WEDNESDAY)
    result = mentor_questions.pending(state, slot)
    offered = [item for item in result.asked if item.kind == "exit_draft_review"]
    assert len(offered) == 2, [item.subject_id for item in offered]

    card = window.trading_panel.alert_center.chart_review.mentor_card
    card.show_slot(slot)
    card.set_questions(result, store=store, service=window.trade_mentor_service)

    for session in (MONDAY, TUESDAY.isoformat()):
        key = check.exit_key(trade_id, session)
        assert card.exit_confirm_button(key) is not None, session
        assert session in card.question_prompt_text("exit_draft_review", key)
        assert f"({session})" in card.question_prompt_text("exit_draft_review", key)
    # And a caller that knows only the trade is REFUSED rather than guessing.
    assert card.draft_key(trade_id) == ""


def test_each_confirm_signs_off_its_own_session_words(tmp_path, monkeypatch, window):
    """Hand-counted: 2 confirms, 2 rows, each citing ITS OWN note and session.

    The writer refuses a note that does not belong to the (trade, session) it
    is asked to confirm, so this cannot pass by accident.
    """
    import trade_mentor_trade_check as check

    store, trade_id, root = _two_session_exit_drafted(tmp_path)
    _point_the_lane_at(monkeypatch, root)
    lane = {row["exit_session"]: row for row in
            window._mentor_exit_drafts(store, WEDNESDAY.isoformat())}

    for draft in lane.values():
        result = check.confirm_exit_fields(
            store, trade_id, draft,
            now=datetime.fromisoformat("2026-09-16T09:05:00-04:00"),
        )
        assert result["ok"] is True, result

    rows = store.list_opportunity_events(
        trade_id=trade_id, event_type=check.EVENT_EXIT_FIELDS, limit=100
    )
    assert len(rows) == 2, rows
    by_session = {row["payload"]["exit_session"]: row["payload"] for row in rows}
    assert set(by_session) == {MONDAY, TUESDAY.isoformat()}
    notes = {note["exit_session"]: note["note_id"] for note in check.exit_notes(store, trade_id)}
    for session, payload in by_session.items():
        assert payload["note_id"] == notes[session], (session, payload)


def test_the_writer_refuses_a_note_from_the_other_session(tmp_path, monkeypatch, window):
    """The guard itself. Hand-counted: 1 refusal, 0 rows written."""
    import trade_mentor_trade_check as check

    store, trade_id, root = _two_session_exit_drafted(tmp_path)
    _point_the_lane_at(monkeypatch, root)
    lane = {row["exit_session"]: row for row in
            window._mentor_exit_drafts(store, WEDNESDAY.isoformat())}

    crossed = dict(lane[MONDAY])
    crossed["note_id"] = lane[TUESDAY.isoformat()]["note_id"]
    result = check.confirm_exit_fields(
        store, trade_id, crossed,
        now=datetime.fromisoformat("2026-09-16T09:05:00-04:00"),
    )
    assert result["ok"] is False, result
    assert "not an exit note" in str(result["reason"]), result
    assert store.list_opportunity_events(
        trade_id=trade_id, event_type=check.EVENT_EXIT_FIELDS, limit=100
    ) == []


def test_over_budget_the_second_reading_is_carried_and_said(tmp_path, monkeypatch, window):
    """What is over budget is COUNTED and CARRIED, never dropped - including
    when both readings belong to ONE trade. Hand-counted: 4 readings on 2
    trades, 3 asked, 1 carried and named."""
    import mentor_questions

    store, trade_id, root = _two_session_exit_drafted(tmp_path)
    other = _day_trade(store, "OTHR", MONDAY)
    import trade_mentor_trade_check as check

    check.save_exit_note(
        store, other, fx.EXIT_NOTE, exit_session=MONDAY,
        now=datetime.fromisoformat("2026-09-15T09:30:00-04:00"),
    )
    _day_trade(store, "THRD", TUESDAY.isoformat())
    third = str(
        [row for row in store.list_trades(trade_date=TUESDAY.isoformat())
         if str(row.get("symbol")) == "THRD"][0]["trade_id"]
    )
    check.save_exit_note(
        store, third, fx.EXIT_NOTE, exit_session=TUESDAY.isoformat(),
        now=datetime.fromisoformat("2026-09-15T09:40:00-04:00"),
    )
    _night(store, MONDAY, root)
    _night(store, TUESDAY.isoformat(), root)
    _point_the_lane_at(monkeypatch, root)

    lane = window._mentor_exit_drafts(store, WEDNESDAY.isoformat())
    assert len(lane) == 4, [(row["symbol"], row["exit_session"]) for row in lane]

    slot, state = _state(window, store, WEDNESDAY)
    result = mentor_questions.pending(state, slot)
    assert len(result.asked) == mentor_questions.BUDGET, result.asked
    seen = {
        item.subject_id
        for item in (*result.asked, *result.carried)
        if item.kind == "exit_draft_review"
    }
    # EVERY reading is either asked or carried. Before this the second reading
    # of one trade was in neither: it was dropped by a de-duplication on the
    # trade id and nothing anywhere counted it.
    assert seen == {row["key"] for row in lane}, sorted(seen)
    assert "exit reading" in result.waiting_note.lower(), result.waiting_note


# ---------------------------------------------------------------------------
# review 2 blocker 2 - a waiting reading has ONE home
# ---------------------------------------------------------------------------
def _widget_counts(card) -> dict[str, int]:
    """Confirm buttons, draft lines and "You wrote:" labels in the WHOLE tree."""
    from PySide6.QtWidgets import QLabel, QPushButton

    confirms = [
        button for button in card.findChildren(QPushButton) if button.text() == "Confirm"
    ]
    labels = [label.text() for label in card.findChildren(QLabel)]
    return {
        "confirms": len(confirms),
        "draft_lines": len([text for text in labels if text.startswith("The night read")]),
        "you_wrote": len([text for text in labels if text.startswith("You wrote:")]),
    }


def _card_with_an_entry_gap(tmp_path, monkeypatch, window):
    """One trade with FOUR open entry fields and a waiting reading on its exit.

    The morning the trader most often meets: a trade they did not finish
    answering, and the night has read the exit note they did write.
    """
    store, ids, root = _monday_exit_drafted(tmp_path)
    _point_the_lane_at(monkeypatch, root)
    slot, state = _state(window, store, TUESDAY)
    card = window.trading_panel.alert_center.chart_review.mentor_card
    card.show_slot(slot)
    return card, store, ids["DAYT"], root, slot, state


def test_a_reading_is_drawn_exactly_once_in_either_call_order(
    tmp_path, monkeypatch, window
):
    """Review 2 blocker 2. Hand-counted: 1 Confirm, 1 draft line, 1 "You
    wrote:" after EVERY order - questions then section, section then questions,
    and a merge of a fresh task on top.

    Before this both seams drew the reading into the SAME bookkeeping slots, so
    the trader saw two copies of the question and the first copy's Confirm
    stopped responding.
    """
    import mentor_questions
    import trade_mentor_trade_check as check

    card, store, trade_id, root, slot, state = _card_with_an_entry_gap(
        tmp_path, monkeypatch, window
    )
    task = check.build_task(store, TUESDAY)
    result = mentor_questions.pending(state, slot)

    card.set_questions(result, store=store, service=window.trade_mentor_service)
    assert _widget_counts(card) == {"confirms": 1, "draft_lines": 1, "you_wrote": 1}

    card.set_trade_check(task, store=store, drafts_root=root)
    assert _widget_counts(card) == {"confirms": 1, "draft_lines": 1, "you_wrote": 1}

    # A fresh task on the next slot of the same session MERGES, and still one.
    card.set_trade_check(check.build_task(store, TUESDAY), store=store, drafts_root=root)
    assert _widget_counts(card) == {"confirms": 1, "draft_lines": 1, "you_wrote": 1}


def test_the_section_alone_still_reaches_the_reading_exactly_once(
    tmp_path, monkeypatch, window
):
    """The other order: a caller that builds only the trade section.

    The reading still has ONE home - the questions block - and the trade
    section says nothing about that exit, because the words are already on the
    card above.
    """
    import trade_mentor_trade_check as check

    card, store, trade_id, root, _slot, _state_payload = _card_with_an_entry_gap(
        tmp_path, monkeypatch, window
    )
    card.set_trade_check(check.build_task(store, TUESDAY), store=store, drafts_root=root)

    assert _widget_counts(card) == {"confirms": 1, "draft_lines": 1, "you_wrote": 1}
    key = check.exit_key(trade_id, MONDAY)
    assert card.exit_confirm_button(key) is not None
    assert card.exit_note_box(trade_id) is None, "the answered exit was asked again"


def test_clearing_the_trade_section_never_drops_a_write_in_flight(
    tmp_path, monkeypatch, window
):
    """Review 2 advisory 4, which blocker 2's one-home rule removes.

    Hand-counted: a Confirm in flight, `_clear_trade_check()`, and the flag is
    STILL True - the reading's row is not part of the trade section and its
    write is not the section's to forget.
    """
    import threading
    import time

    import trade_mentor_trade_check as check

    card, store, trade_id, root, _slot, _payload = _card_with_an_entry_gap(
        tmp_path, monkeypatch, window
    )
    card.set_trade_check(check.build_task(store, TUESDAY), store=store, drafts_root=root)
    key = check.exit_key(trade_id, MONDAY)

    started = threading.Event()
    release = threading.Event()
    real = check.confirm_exit_fields

    def _slow(*args, **kwargs):
        started.set()
        assert release.wait(20.0)
        return real(*args, **kwargs)

    monkeypatch.setattr(check, "confirm_exit_fields", _slow)
    card.exit_confirm_button(key).click()
    assert started.wait(20.0)

    card._clear_trade_check()
    assert card.exit_write_in_flight(key) is True, "the in-flight write was forgotten"

    release.set()
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and card.exit_write_in_flight(key):
        _app.processEvents()
    assert card.exit_write_in_flight(key) is False


# ---------------------------------------------------------------------------
# review 3 - the REGISTRY decides which readings and how many, always
# ---------------------------------------------------------------------------
def _four_unanswered_day_trades(tmp_path, monkeypatch):
    """Four trades closed Monday, entry fields never answered, a note on each.

    The reviewer's reproduction. `trade_origin` fires for all four at priority
    20, so the budget of three is spent before an exit reading is reached and
    the registry offers NONE of them - while the card's own note says four are
    waiting.
    """
    store, ids, root = _monday_exit_drafted(
        tmp_path, symbols=("AAAA", "BBBB", "CCCC", "DDDD")
    )
    _point_the_lane_at(monkeypatch, root)
    return store, ids, root


def test_the_section_never_draws_more_readings_than_the_registry_offered(
    tmp_path, monkeypatch, window
):
    """THE BLOCKER. Hand-counted: the registry offers 0 exit readings here, so
    the card carries 0 - and its own waiting note stays true.

    Before this the trade section drew its own list: 4 Confirms on a card whose
    sentence said four were still waiting, and none of them had been offered.
    """
    import mentor_questions
    import trade_mentor_trade_check as check

    store, _ids, root = _four_unanswered_day_trades(tmp_path, monkeypatch)
    slot, state = _state(window, store, TUESDAY)
    result = mentor_questions.pending(state, slot)
    offered = [item for item in result.asked if item.kind == "exit_draft_review"]
    assert offered == [], [item.subject_id for item in offered]
    assert "exit reading" in result.waiting_note.lower(), result.waiting_note

    card = window.trading_panel.alert_center.chart_review.mentor_card
    card.show_slot(slot)
    card.set_questions(result, store=store, service=window.trade_mentor_service)
    card.set_trade_check(
        check.build_task(store, TUESDAY), store=store, drafts_root=root
    )

    assert _widget_counts(card)["confirms"] == 0, _widget_counts(card)
    assert "exit reading" in card.questions_label.text().lower(), (
        card.questions_label.text()
    )


def test_the_section_alone_asks_the_registry_and_keeps_the_budget(
    tmp_path, monkeypatch, window
):
    """A caller that builds ONLY the trade section still obeys the budget.

    Hand-counted: 4 readings waiting, 3 rows drawn OLDEST FIRST, and the card
    SAYS the fourth is still waiting. It asks `mentor_questions.pending` - the
    same function, the same budget, the same ordering - never a list of its
    own.
    """
    import mentor_questions
    import trade_mentor_trade_check as check

    store, ids, root = _four_unanswered_day_trades(tmp_path, monkeypatch)
    slot, _state_payload = _state(window, store, TUESDAY)

    card = window.trading_panel.alert_center.chart_review.mentor_card
    card.show_slot(slot)
    card.set_trade_check(
        check.build_task(store, TUESDAY), store=store, drafts_root=root
    )

    assert mentor_questions.BUDGET == 3
    counts = _widget_counts(card)
    assert counts["confirms"] == 3, counts
    assert counts["draft_lines"] == 3, counts
    drawn = [
        key for key in (check.exit_key(ids[symbol], MONDAY)
                        for symbol in ("AAAA", "BBBB", "CCCC", "DDDD"))
        if card.exit_confirm_button(key) is not None
    ]
    assert drawn == [check.exit_key(ids[symbol], MONDAY)
                     for symbol in ("AAAA", "BBBB", "CCCC")], drawn
    assert "1" in card.questions_label.text(), card.questions_label.text()
    assert "exit reading" in card.questions_label.text().lower()


def test_away_draws_no_reading_through_the_section_seam_either(
    tmp_path, monkeypatch, window
):
    """AWAY prompts nothing, wherever the question came from.

    The rule lives in the registry, and the section seam asks the registry -
    so it inherits it rather than keeping a second copy. Hand-counted: 4
    readings waiting, 0 rows.
    """
    import trade_mentor_trade_check as check

    store, _ids, root = _four_unanswered_day_trades(tmp_path, monkeypatch)
    slot, _payload = _state(window, store, TUESDAY, auto_mode="AWAY")

    card = window.trading_panel.alert_center.chart_review.mentor_card
    card.show_slot(slot)
    card.set_trade_check(
        check.build_task(store, TUESDAY), store=store, drafts_root=root,
        auto_mode="AWAY",
    )

    assert _widget_counts(card)["confirms"] == 0, _widget_counts(card)


def test_both_call_orders_end_with_the_registrys_own_answer(
    tmp_path, monkeypatch, window
):
    """Order independence, over the case that used to break it.

    Hand-counted: questions-then-section and section-then-questions both end
    at the registry's answer - 0 readings here - and the same task twice
    changes nothing.
    """
    import mentor_questions
    import trade_mentor_trade_check as check

    store, _ids, root = _four_unanswered_day_trades(tmp_path, monkeypatch)
    slot, state = _state(window, store, TUESDAY)
    result = mentor_questions.pending(state, slot)
    task = check.build_task(store, TUESDAY)
    card = window.trading_panel.alert_center.chart_review.mentor_card

    card.show_slot(slot)
    card.set_questions(result, store=store, service=window.trade_mentor_service)
    card.set_trade_check(task, store=store, drafts_root=root)
    forwards = _widget_counts(card)

    card.show_slot(slot)
    card.set_trade_check(task, store=store, drafts_root=root)
    card.set_questions(result, store=store, service=window.trade_mentor_service)
    backwards = _widget_counts(card)

    card.set_trade_check(check.build_task(store, TUESDAY), store=store, drafts_root=root)
    again = _widget_counts(card)

    assert forwards == backwards == again, (forwards, backwards, again)
    assert forwards["confirms"] == 0, forwards


def test_the_next_card_offers_the_readings_once_the_origin_questions_are_answered(
    tmp_path, monkeypatch, window
):
    """The morning after: the budget frees up and the readings arrive.

    Hand-counted: with the four origin questions retired, the registry offers
    3 readings and says 1 more is waiting - which is the desk's own rule about
    what is over budget, now reaching the readings too.
    """
    import mentor_questions

    store, _ids, root = _four_unanswered_day_trades(tmp_path, monkeypatch)
    slot, state = _state(window, store, TUESDAY)
    payload = dict(state)
    # The trader answered where those four trades came from yesterday, so that
    # kind asks nothing today. Nothing else about the card changes.
    payload["retired"] = tuple(
        f"trade_origin:{subject.subject_id}"
        for subject in mentor_questions._trigger_trade_origin(payload)
    )

    result = mentor_questions.pending(payload, slot)
    offered = [item for item in result.asked if item.kind == "exit_draft_review"]
    carried = [item for item in result.carried if item.kind == "exit_draft_review"]
    assert len(offered) == 3, [item.subject_id for item in offered]
    assert len(carried) == 1, [item.subject_id for item in carried]

    card = window.trading_panel.alert_center.chart_review.mentor_card
    card.show_slot(slot)
    card.set_questions(result, store=store, service=window.trade_mentor_service)
    assert _widget_counts(card)["confirms"] == 3
    assert "1" in card.questions_label.text(), card.questions_label.text()


# ---------------------------------------------------------------------------
# review 2 advisories 2 and 3 - rewriting, and a reading of words that changed
# ---------------------------------------------------------------------------
def test_a_superseded_reading_is_not_offered_and_the_night_reads_the_new_words(
    tmp_path, monkeypatch, window
):
    """Review 2 advisory 3. Hand-counted through the whole sequence:
    note -> night -> 1 offered; second note -> 0 offered; night -> 1 offered,
    and it is a reading of the NEW words."""
    import trade_mentor_trade_check as check

    store, ids, root = _monday_exit_drafted(tmp_path)
    _point_the_lane_at(monkeypatch, root)
    trade_id = ids["DAYT"]
    assert len(window._mentor_exit_drafts(store, WEDNESDAY.isoformat())) == 1

    check.save_exit_note(
        store, trade_id, fx.SECOND_EXIT_NOTE, exit_session=MONDAY,
        now=datetime.fromisoformat("2026-09-15T10:00:00-04:00"),
    )
    assert window._mentor_exit_drafts(store, WEDNESDAY.isoformat()) == [], (
        "a reading of words the trader rewrote was still offered"
    )

    # The night reads the NEW note: its already-drafted check is per note_id.
    from ai_jobs import exit_note_fields

    waiting = exit_note_fields.notes_waiting(store, MONDAY, root=root)
    assert [note["text"] for note in waiting] == [fx.SECOND_EXIT_NOTE], waiting
    out = exit_note_fields.run_exit_note_fields(
        session_date=MONDAY, now=TUESDAY_NIGHT, root=root, store=store,
        request=fx.fake_request(
            {"fields": {"why": fx.value(
                "I needed the capital",
                __import__("exit_reasons").codes()[1],
                text=fx.SECOND_EXIT_NOTE,
            )}}
        ),
    )
    assert out["status"] == "ok", out
    lane = window._mentor_exit_drafts(store, WEDNESDAY.isoformat())
    assert len(lane) == 1, lane
    assert lane[0]["raw_text"] == fx.SECOND_EXIT_NOTE


def test_rewrite_appends_a_superseding_note_and_leaves_the_first_alone(
    tmp_path, monkeypatch, window
):
    """Review 2 advisory 2: the supersede path finally has a door.

    Hand-counted: 2 `EXIT_NOTE_RAW` rows afterwards, row 0 byte-identical, and
    the reading of the old words gone from the lane.
    """
    import time

    import mentor_questions
    import trade_mentor_trade_check as check

    card, store, trade_id, root, slot, state = _card_with_an_entry_gap(
        tmp_path, monkeypatch, window
    )
    card.set_questions(
        mentor_questions.pending(state, slot),
        store=store,
        service=window.trade_mentor_service,
    )
    key = check.exit_key(trade_id, MONDAY)
    before = [dict(row) for row in store.list_opportunity_events(
        trade_id=trade_id, event_type=check.EVENT_EXIT_NOTE_RAW, limit=100
    )]
    assert len(before) == 1

    assert card.exit_rewrite_box(key).isVisibleTo(card) is False
    card.exit_rewrite_button(key).click()
    assert card.exit_rewrite_box(key).toPlainText() == fx.EXIT_NOTE
    card.exit_rewrite_box(key).setPlainText(fx.SECOND_EXIT_NOTE)
    card._save_exit_rewrite(key)

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and card.exit_write_in_flight(key):
        _app.processEvents()

    after = store.list_opportunity_events(
        trade_id=trade_id, event_type=check.EVENT_EXIT_NOTE_RAW, limit=100
    )
    assert len(after) == 2, after
    assert dict(after[0]) == before[0], "the first note was rewritten"
    assert after[1]["payload"]["raw_text"] == fx.SECOND_EXIT_NOTE
    assert window._mentor_exit_drafts(store, WEDNESDAY.isoformat()) == []


def test_rewrite_left_unchanged_writes_nothing(tmp_path, monkeypatch, window):
    """Clicking Rewrite and thinking better of it is not a new note.

    Hand-counted: 1 `EXIT_NOTE_RAW` row before and after, on both seams.
    """
    import mentor_questions
    import trade_mentor_trade_check as check

    card, store, trade_id, root, slot, state = _card_with_an_entry_gap(
        tmp_path, monkeypatch, window
    )
    card.set_questions(
        mentor_questions.pending(state, slot),
        store=store,
        service=window.trade_mentor_service,
    )
    key = check.exit_key(trade_id, MONDAY)
    card.exit_rewrite_button(key).click()
    assert card._save_exit_rewrite(key) is False
    assert "nothing" in card.status_label.text().lower(), card.status_label.text()
    assert len(store.list_opportunity_events(
        trade_id=trade_id, event_type=check.EVENT_EXIT_NOTE_RAW, limit=100
    )) == 1


# ---------------------------------------------------------------------------
# the budget, the window, AWAY, and what a draft row is NOT
# ---------------------------------------------------------------------------
def test_three_readings_are_offered_and_the_fourth_is_said_and_carried(
    tmp_path, monkeypatch, window
):
    """The budget of three is the budget of three. Hand-counted: 4 waiting ->
    3 asked OLDEST FIRST, 1 carried, and the card SAYS so."""
    import mentor_questions

    store, ids, root = _monday_exit_drafted(
        tmp_path, symbols=("AAAA", "BBBB", "CCCC", "DDDD")
    )
    _point_the_lane_at(monkeypatch, root)

    lane = window._mentor_exit_drafts(store, WEDNESDAY.isoformat())
    assert [row["symbol"] for row in lane] == ["AAAA", "BBBB", "CCCC", "DDDD"], lane

    slot, state = _state(window, store, WEDNESDAY)
    result = mentor_questions.pending(state, slot)
    asked = [item for item in result.asked if item.kind == "exit_draft_review"]
    carried = [item for item in result.carried if item.kind == "exit_draft_review"]
    assert [item.detail["symbol"] for item in asked] == ["AAAA", "BBBB", "CCCC"]
    assert [item.detail["symbol"] for item in carried] == ["DDDD"]
    assert mentor_questions.BUDGET == 3
    assert "exit reading" in result.waiting_note.lower(), result.waiting_note
    assert "1" in result.waiting_note, result.waiting_note


def test_a_reading_past_the_window_is_not_offered_and_stays_provisional(
    tmp_path, monkeypatch, window
):
    """Nothing is ever confirmed by AGE. Hand-counted: 6 sessions on, 0
    offered, and the draft still on disk reading `provisional`."""
    import journal_store
    import trade_mentor_trade_check as check
    from ai_jobs import exit_note_fields

    store, ids, root = _monday_exit_drafted(tmp_path)
    _point_the_lane_at(monkeypatch, root)
    trade_id = ids["DAYT"]

    window_days = check.offer_window(
        date(2026, 9, 23).isoformat(), check.EXIT_DRAFT_OFFER_SESSIONS
    )
    assert MONDAY not in window_days, window_days

    assert window._mentor_exit_drafts(store, "2026-09-23") == []
    stored = exit_note_fields.draft_for(trade_id, MONDAY, root=root)
    assert stored["status"] == journal_store.TAG_STATUS_PROVISIONAL
    assert check.exit_fields(store, trade_id) == {}


def test_away_offers_no_reading_at_all(tmp_path, monkeypatch, window):
    """AWAY prompts nothing - forced rows included, and a reading is not more
    urgent than the trader not being there."""
    import mentor_questions

    store, _ids, root = _monday_exit_drafted(tmp_path)
    _point_the_lane_at(monkeypatch, root)

    slot, state = _state(window, store, WEDNESDAY, auto_mode="AWAY")
    result = mentor_questions.pending(state, slot)
    assert result.asked == ()
    assert result.forced == ()


def test_a_waiting_reading_is_neither_an_unlabelled_trade_nor_an_open_exit(
    tmp_path, monkeypatch, window
):
    """A draft row is something to LOOK at. Hand-counted: 0 unlabelled, 0
    unexplained exits, and Save is untouched by the row being on the card."""
    import mentor_questions
    import trade_mentor_trade_check as check

    store, ids, root = _monday_exit_drafted(tmp_path)
    _point_the_lane_at(monkeypatch, root)

    # The trade's four ENTRY fields are still open - that is TJ-9's question
    # and it is untouched. Its EXIT is explained, so the exit count is 0 and a
    # waiting reading adds nothing to either number.
    assert check.unlabelled_trade_count(store, MONDAY) == 1
    assert check.unexplained_exit_count(store, MONDAY) == 0

    slot, state = _state(window, store, WEDNESDAY)
    card = window.trading_panel.alert_center.chart_review.mentor_card
    card.show_slot(slot)
    card.set_trade_check(check.build_task(store, WEDNESDAY), store=store, drafts_root=root)
    before = card.save_answers_button.isEnabled()
    card.set_questions(
        mentor_questions.pending(state, slot),
        store=store,
        service=window.trade_mentor_service,
    )
    assert card.save_answers_button.isEnabled() is before, "a reading moved the gate"
