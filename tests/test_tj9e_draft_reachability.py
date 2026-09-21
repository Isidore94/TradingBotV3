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

    import mentor_questions

    slot, state = _state(window, store, WEDNESDAY)
    result = mentor_questions.pending(state, slot)
    offered = [item for item in result.asked if item.kind == "exit_draft_review"]
    assert len(offered) == 1, [item.kind for item in result.asked]

    card = window.trading_panel.alert_center.chart_review.mentor_card
    card.show_slot(slot)
    card.set_questions(result, store=store, service=window.trade_mentor_service)

    assert "DAYT" in card.question_prompt_text("exit_draft_review", trade_id)
    assert MONDAY in card.question_prompt_text("exit_draft_review", trade_id)
    assert card.exit_draft_line(trade_id).strip(), "no draft line was drawn"
    assert card.exit_confirm_button(trade_id) is not None
    assert card.exit_correct_button(trade_id) is not None
    # It is NOT a combo: a reading is confirmed or corrected, never picked.
    assert card.question_box("exit_draft_review", trade_id) is None


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
    assert card.question_box("exit_draft_review", trade_id) is None


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
