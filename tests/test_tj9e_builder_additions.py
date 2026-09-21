r"""TJ-9E - what the tester did not pin, and what their pins could not say here.

TWO GAPS the packet named and the tester left to the builder, each with its own
fail-before-fix proof recorded in the commit that introduced it:

1. **the status vocabulary has ONE owner** (the packet's CORRECTED point 2).
   The tester pinned the BEHAVIOUR - a `CLOSED_PARTIAL` trade is listed - and
   deliberately asserted no constant. This pins the other half: every status
   the assembler really stamps is a status the morning check really accepts, so
   a fourth status added to `journal_store` later cannot silently drop a trade
   off the 09:00 card the way `PARTIALLY_CLOSED` dropped seven live ones.
2. **a trade whose exits fall in TWO sessions is asked on BOTH mornings**
   (the packet's CORRECTED point 6; 13 live trades have closing legs on two
   dates). One box per (trade, exit session), two `EXIT_NOTE_RAW` rows with
   different `exit_session` values, each drafted and confirmed separately.

And FOUR guarantees whose tester-written pins cannot pass on this desk for
reasons that belong to the pin rather than to the code. None of those four
tests was edited - the builder reported each with its measurement and left it
red - and none of the four guarantees is allowed to go unpinned because of
that, so each is re-stated below in the form that can be true. Every one of
them names the tester's test and what is wrong with it.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
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

#: The Tuesday after the fixture's Monday. Its previous exchange session is the
#: Monday, which is where the second half of the two-session exit lands.
SECOND_SESSION = "2026-09-14"
SECOND_MORNING = date(2026, 9, 15)


def test_every_status_the_assembler_writes_is_one_the_morning_check_accepts(tmp_path):
    """The two lists are ONE list. Hand-counted: 3 statuses, 3 accepted.

    Asserted against the ASSEMBLED rows as well as the constant, so a status
    that is renamed in `journal_store` and not re-exported fails here rather
    than quietly emptying a morning card.
    """
    import journal_store
    import trade_mentor_trade_check as check

    store, ids = fx.ready_store(tmp_path)

    seen = {
        str(store.get_trade(trade_id).get("status") or "")
        for trade_id in ids.values()
    }
    assert seen == {
        journal_store.TRADE_STATUS_CLOSED,
        journal_store.TRADE_STATUS_CLOSED_PARTIAL,
        journal_store.TRADE_STATUS_OPEN,
    }, sorted(seen)
    assert set(check._SESSION_STATUSES) == set(journal_store.TRADE_STATUSES)
    assert not seen - set(check._SESSION_STATUSES), sorted(seen)


def _two_session_exit(tmp_path):
    """ONE trade, opened Friday, HALF exited Friday and the rest on Monday.

    Hand-counted: 3 executions, 2 closing legs, on 2 different dates - the live
    shape of 13 of the journal's 216 trades.
    """
    from journal_importers import manual_execution_from_fields

    def _row(execution_id, side, qty, price, stamp):
        return manual_execution_from_fields(
            {
                "broker": "MANUAL",
                "account_number": fx.ACCOUNT,
                "symbol": fx.SCALE_OUT,
                "side": side,
                "quantity": qty,
                "price": price,
                "timestamp": stamp,
                "security_type": "STK",
                "currency": "USD",
                "commission": 0,
                "fees": 0,
                "execution_id": execution_id,
            }
        )

    store = fx.new_store(tmp_path)
    fx.mark_covered(store, fx.REVIEWED)
    fx.mark_covered(store, SECOND_SESSION)
    store.upsert_executions(
        [
            _row("TWO-1", "BUY", 100, 40.0, f"{fx.REVIEWED}T07:31:00"),
            _row("TWO-2", "SELL", 40, 41.0, f"{fx.REVIEWED}T10:05:00"),
            _row("TWO-3", "SELL", 60, 43.0, f"{SECOND_SESSION}T10:05:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    trades = store.list_trades(trade_date=SECOND_SESSION)
    assert len(trades) == 1, trades
    return store, str(trades[0]["trade_id"])


def test_a_trade_that_exits_in_two_sessions_is_asked_on_both_mornings(tmp_path):
    """Hand-counted: ONE exit box on Monday for Friday's scale, ONE on Tuesday
    for Monday's - and each names its OWN session.

    "Asked once per trade per session" cuts both ways: two closing legs in one
    session are one question (the tester pinned that), and two closing legs in
    two sessions are two. A trade that left half the position on Friday and the
    rest on Monday was two decisions, and merging them would ask the trader to
    explain both in one box, on one of the two mornings, about whichever exit
    the code happened to look at.
    """
    import trade_mentor_trade_check as check

    store, trade_id = _two_session_exit(tmp_path)

    legs = [
        leg for leg in store.list_trade_legs(trade_id)
        if str(leg.get("role") or "").upper() == "CLOSE"
    ]
    assert len(legs) == 2, legs
    assert len({str(leg["timestamp"])[:10] for leg in legs}) == 2, legs

    monday = check.build_task(store, fx.SESSION_TODAY)
    monday_rows = [row for row in monday.trades if row.trade_id == trade_id]
    assert len(monday_rows) == 1, monday_rows
    assert str(monday_rows[0].exit_session) == fx.REVIEWED

    tuesday = check.build_task(store, SECOND_MORNING)
    tuesday_rows = [row for row in tuesday.trades if row.trade_id == trade_id]
    assert len(tuesday_rows) == 1, tuesday_rows
    assert str(tuesday_rows[0].exit_session) == SECOND_SESSION


def test_the_two_notes_are_two_rows_and_each_is_read_back_by_its_own_session(tmp_path):
    """Hand-counted: 2 `EXIT_NOTE_RAW` rows for ONE trade, one per session.

    The session-wide read is what the night and the Day Review both use, so
    each morning's note has to come back under its own session and never both
    under one - otherwise Tuesday's draft would be a reading of Friday's words.
    """
    import trade_mentor_trade_check as check

    store, trade_id = _two_session_exit(tmp_path)

    check.save_exit_note(
        store,
        trade_id,
        fx.EXIT_NOTE,
        exit_session=fx.REVIEWED,
        now=datetime.fromisoformat("2026-09-14T09:05:00-04:00"),
    )
    check.save_exit_note(
        store,
        trade_id,
        fx.SECOND_EXIT_NOTE,
        exit_session=SECOND_SESSION,
        now=datetime.fromisoformat("2026-09-15T09:05:00-04:00"),
    )

    rows = check.exit_notes(store, trade_id)
    assert [row["exit_session"] for row in rows] == [fx.REVIEWED, SECOND_SESSION], rows

    friday = check.exit_notes_for_session(store, fx.REVIEWED)
    monday = check.exit_notes_for_session(store, SECOND_SESSION)
    assert friday[trade_id]["raw_text"] == fx.EXIT_NOTE
    assert monday[trade_id]["raw_text"] == fx.SECOND_EXIT_NOTE
    assert friday[trade_id]["note_id"] != monday[trade_id]["note_id"]


def test_the_night_drafts_each_session_of_a_two_session_exit_on_its_own(tmp_path):
    """Hand-counted: 1 model call per session, and a draft keyed to that
    session alone. The second night must not re-ask the first session's note."""
    import exit_reasons
    import trade_mentor_trade_check as check
    import trader_state_tags
    from ai_jobs import exit_note_fields

    root = tmp_path / "packs"
    store, trade_id = _two_session_exit(tmp_path)
    check.save_exit_note(
        store, trade_id, fx.EXIT_NOTE, exit_session=fx.REVIEWED,
        now=datetime.fromisoformat("2026-09-14T09:05:00-04:00"),
    )
    check.save_exit_note(
        store, trade_id, fx.SECOND_EXIT_NOTE, exit_session=SECOND_SESSION,
        now=datetime.fromisoformat("2026-09-15T09:05:00-04:00"),
    )

    reply = fx.good_reply(exit_reasons.codes()[0], trader_state_tags.codes()[:1])
    # Built by hand against the SECOND note's own words: it says why and
    # nothing about a feeling, so `felt` is absent rather than guessed.
    second = {
        "fields": {
            "why": fx.value(
                "I needed the capital", exit_reasons.codes()[1], text=fx.SECOND_EXIT_NOTE
            ),
            "watching": [
                fx.value("the capital for the open", text=fx.SECOND_EXIT_NOTE)
            ],
        }
    }

    first_calls: list[dict] = []
    out = exit_note_fields.run_exit_note_fields(
        session_date=fx.REVIEWED,
        now=datetime.fromisoformat("2026-09-14T02:00:00-04:00"),
        root=root,
        store=store,
        request=fx.fake_request(reply, calls=first_calls),
    )
    assert out["status"] == "ok", out
    assert len(first_calls) == 1, first_calls

    later_calls: list[dict] = []
    out = exit_note_fields.run_exit_note_fields(
        session_date=SECOND_SESSION,
        now=datetime.fromisoformat("2026-09-15T02:00:00-04:00"),
        root=root,
        store=store,
        request=fx.fake_request(second, calls=later_calls),
    )
    assert out["status"] == "ok", out
    assert len(later_calls) == 1, "the second night re-read the first session's note"

    friday = exit_note_fields.draft_for(trade_id, fx.REVIEWED, root=root)
    monday = exit_note_fields.draft_for(trade_id, SECOND_SESSION, root=root)
    assert friday["fields"]["why"]["code"] == exit_reasons.codes()[0]
    assert monday["fields"]["why"]["code"] == exit_reasons.codes()[1]
    assert friday["note_id"] != monday["note_id"]


# ---------------------------------------------------------------------------
# the two guarantees whose tester-written pins cannot pass on this desk
#
# Both tester lines are reported to the lead as wrong literals and NEITHER was
# edited. Neither guarantee is allowed to go unpinned because of that, so each
# is re-pinned here in the form that can actually be true.
# ---------------------------------------------------------------------------
def test_the_request_body_holds_the_words_and_no_money_serialised_honestly(tmp_path):
    """The outcome fence, on the EXACT body the fake endpoint received.

    The same assertion as
    `test_tj9e_night_slot.py::test_the_request_body_holds_the_words_the_symbol_the_side_and_two_code_lists`,
    with ONE difference: the note is looked for AS IT IS ENCODED.

    The fixture note carries an en dash on purpose, and the shared provider
    path embeds the evidence with `json.dumps` at its default
    ``ensure_ascii=True`` (`ai_summary._local_schema_prompt`), so the prompt
    carries ``\\u2013`` where the note has the dash. ``EXIT_NOTE in
    json.dumps(body)`` is therefore False for every possible implementation -
    including one that sends the trader's words untouched, which is what this
    proves the slot does. The model still SEES the dash: it decodes the JSON
    string it was handed.

    Hand-counted: 1 model call, 0 forbidden keys at any depth, 0 of the four
    money strings anywhere in the body.
    """
    import exit_reasons
    import test_tj9e_night_slot as night_slot
    import trader_state_tags
    from ai_jobs import exit_note_fields

    store, _trade_id = fx.swing_with_money(tmp_path)
    calls: list[dict] = []
    why = exit_reasons.codes()[0]
    felt = trader_state_tags.codes()[0]

    with night_slot._live_settings():
        out = exit_note_fields.run_exit_note_fields(
            session_date=fx.REVIEWED,
            now=night_slot.NIGHT,
            root=tmp_path / "packs",
            store=store,
            post=fx.fake_post(fx.good_reply(why, (felt,)), calls),
        )
    assert out["status"] == "ok", out
    assert len(calls) == 1, calls

    body = calls[0]["json"]
    sent = json.dumps(body, default=str, ensure_ascii=False)
    #: The prompt the model is actually handed. The note is looked for HERE and
    #: in the form the evidence JSON carries it - the dash escaped, every other
    #: character its own. Not a looser check: the same characters, spelled the
    #: way the wire spells them, in the string the model decodes.
    prompt = " ".join(str(row.get("content") or "") for row in body["messages"])

    assert json.dumps(fx.EXIT_NOTE)[1:-1] in prompt, (
        "the model must see the trader's own words"
    )
    assert fx.SWING in sent, "the symbol travels"
    assert "LONG" in sent, "the side travels"
    assert why in sent and felt in sent, "both code lists travel"

    leaked = night_slot._forbidden_keys(body)
    assert leaked == [], leaked
    for number in (
        fx.MONEY_ENTRY_PRICE, fx.MONEY_EXIT_PRICE, fx.MONEY_NET_PNL, fx.MONEY_QUANTITY,
    ):
        assert str(number) not in sent, f"{number} reached the prompt"


def test_a_date_only_exit_note_is_never_a_same_session_note(tmp_path):
    """The live DRAM shape: a closing fill stamped midnight in its OWN offset.

    `test_tj9e_row_shapes.py::test_a_date_only_exit_is_asked_and_its_note_is_never_same_session`
    asserts `journal_trade_shape.is_date_only` is True for its fixture's
    closing leg. That fill is written through `manual_execution_from_fields`,
    which attaches the DESK's local zone to a naive stamp - Pacific here - so
    the stored row is ``2026-09-11T00:00:00-07:00`` and that function, which
    asks the MARKET-LOCAL question alone, reads it as a fill at 03:00 in New
    York. The tester's line is therefore true only on an Eastern desk. It is
    reported rather than edited; this pins the BEHAVIOUR it was guarding, over
    exactly the shape the live journal holds (`trade_origin._is_midnight`'s own
    docstring records the DRAM row at ``2026-07-16T00:00:00-07:00``).

    Hand-counted: 1 trade, 1 date-only closing leg, and a note typed on the
    exit's own date that is still NOT `same_session`.
    """
    import trade_mentor_trade_check as check
    import trade_origin
    from journal_importers import manual_execution_from_fields

    def _row(execution_id, side, qty, price, stamp):
        return manual_execution_from_fields(
            {
                "broker": "MANUAL",
                "account_number": fx.ACCOUNT,
                "symbol": fx.DATE_ONLY_EXIT,
                "side": side,
                "quantity": qty,
                "price": price,
                "timestamp": stamp,
                "security_type": "STK",
                "currency": "USD",
                "commission": 0,
                "fees": 0,
                "execution_id": execution_id,
            }
        )

    store = fx.new_store(tmp_path)
    fx.mark_covered(store, fx.REVIEWED)
    store.upsert_executions(
        [
            _row("DO-1", "BUY", 100, 60.0, f"{fx.PRIOR}T08:15:00-04:00"),
            # Midnight in its OWN offset, which is how the desk really stores a
            # broker row that carries no clock time.
            _row("DO-2", "SELL", 100, 61.0, f"{fx.REVIEWED}T00:00:00-07:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    trade_id = str(store.list_trades(trade_date=fx.REVIEWED)[0]["trade_id"])

    written = check.save_exit_note(
        store,
        trade_id,
        fx.EXIT_NOTE,
        exit_session=fx.REVIEWED,
        # Typed on the exit's OWN date - the only moment a naive rule could
        # call `same_session`.
        now=datetime.fromisoformat(f"{fx.REVIEWED}T11:00:00-04:00"),
    )
    payload = written["payload"]
    assert payload["label_provenance"] == trade_origin.RECALLED_AFTER, payload
    assert payload["label_provenance_reason"] == check.REASON_DATE_ONLY_FILL, payload
    assert payload["written_after_the_session"] is True, payload


def _card(tmp_path):
    from ui.widgets.trade_mentor_card import TradeMentorCard

    return TradeMentorCard(drafts_path=Path(tmp_path) / "drafts.json")


def _swing_with_a_draft(tmp_path):
    """One swing, entry ANSWERED, one exit note, and one night's draft.

    `test_tj9e_draft_is_not_the_traders.py::test_the_draft_line_never_greys_save`
    says "the swing's exit box is the only gate" - but its `_drafted` helper
    builds the trade through `tj9e_support.swing_with_money`, which never
    answers the four material fields, so that card really carries four
    unanswered entry combos and Save is grey for TJ-9's own forced reason. The
    fixture here answers the entry, which is the state the sentence describes.
    """
    import exit_reasons
    import trade_mentor_trade_check as check
    import trader_state_tags
    from ai_jobs import exit_note_fields

    store, trade_id = fx.swing_only(tmp_path)
    check.save_exit_note(
        store,
        trade_id,
        fx.EXIT_NOTE,
        exit_session=fx.REVIEWED,
        now=datetime.fromisoformat("2026-09-14T09:05:00-04:00"),
    )
    root = tmp_path / "packs"
    out = exit_note_fields.run_exit_note_fields(
        session_date=fx.REVIEWED,
        now=datetime.fromisoformat("2026-09-14T02:00:00-04:00"),
        root=root,
        store=store,
        request=fx.fake_request(
            fx.good_reply(exit_reasons.codes()[0], trader_state_tags.codes()[:1])
        ),
    )
    assert out["status"] == "ok", out
    return store, trade_id, root


def test_a_waiting_draft_never_greys_save(tmp_path):
    """A waiting draft is something to LOOK at, not a field to fill.

    Hand-counted: ONE row, no entry gap, so the exit box is the whole gate.
    Grey with the box empty, green once words are typed - with a draft waiting
    the whole time. A reading nobody clicked must never hold the morning
    hostage.
    """
    import trade_mentor_trade_check as check

    store, trade_id, root = _swing_with_a_draft(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)
    assert len(task.trades) == 1, [(q.symbol, q.missing) for q in task.trades]
    assert task.trades[0].missing == (), task.trades[0].missing

    card = _card(tmp_path)
    card.set_trade_check(task, store=store, drafts_root=root)
    assert card.exit_draft_line(trade_id).strip(), "no draft line was shown"
    assert card.save_answers_button.isEnabled() is False, "nothing written yet"

    card.exit_note_box(trade_id).setPlainText(fx.EXIT_NOTE)
    assert card.save_answers_button.isEnabled() is True, "a waiting draft greyed Save"


def test_one_write_is_in_flight_and_every_ending_settles_the_buttons(tmp_path, monkeypatch):
    """Nothing expensive on the Qt thread, no double write, and a DEADLINE.

    `test_tj9e_draft_is_not_the_traders.py::test_one_write_is_in_flight_and_every_ending_re_enables_the_buttons`
    waits for the write with a fixed 200 turns of `processEvents`. Measured on
    this desk: 200 of them cost 0.29 ms and ONE `record_opportunity_event`
    costs 6.9 ms, because it is a real SQLite commit - so that budget is short
    by a factor of about twenty for any implementation that writes the journal
    off the Qt thread, which is the thing the test is there to require. The
    house rule is that every wait carries a DEADLINE (CLAUDE.md), so this one
    does.

    Hand-counted: two clicks while one write is in flight -> exactly ONE call,
    both verbs grey for the whole of it, both settled afterwards, and the row
    on disk.
    """
    import trade_mentor_trade_check as check

    store, trade_id, root = _swing_with_a_draft(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)

    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []
    real_confirm = check.confirm_exit_fields

    def _slow_confirm(*args, **kwargs):
        calls.append("confirm")
        started.set()
        assert release.wait(20.0), "the writer was never released"
        return real_confirm(*args, **kwargs)

    monkeypatch.setattr(check, "confirm_exit_fields", _slow_confirm)

    card = _card(tmp_path)
    card.set_trade_check(task, store=store, drafts_root=root)
    confirm = card.exit_confirm_button(trade_id)
    correct = card.exit_correct_button(trade_id)

    confirm.click()
    assert started.wait(20.0), "the confirm never reached a worker"
    assert confirm.isEnabled() is False, "Confirm stayed clickable mid-write"
    assert correct.isEnabled() is False, "Correct stayed clickable mid-write"

    confirm.click()  # the second click must find nothing to start
    assert len(calls) == 1, f"{len(calls)} writes in flight"

    release.set()
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and card.exit_write_in_flight(trade_id):
        _app.processEvents()
    assert card.exit_write_in_flight(trade_id) is False, "the card never settled"
    assert confirm.isEnabled() is True
    assert correct.isEnabled() is True
    assert check.exit_fields(store, trade_id)["fields"]["why"]["code"]
