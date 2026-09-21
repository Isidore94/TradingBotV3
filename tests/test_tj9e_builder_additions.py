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

FOUR guarantees whose tester-written pins could not pass on this desk for
reasons that belonged to the pin rather than to the code. The builder edited
none of them and reported each with its measurement; the LEAD then amended all
four (2026-09-21) and the reviewer judged every amendment sound. The re-pins
below stay: they say the same things in a second shape, and two of them are
now the only green statement of their guarantee.

And FIVE more, one per blocker of review round 1 - the shared-store seam, the
report card's wiring, the ride, an answered exit, and the money fence's own
value allow-list. Each names the blocker it answers.
"""

from __future__ import annotations

import json
import os
import re
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
    # LEAD-GRANTED AMENDMENT 2026-09-21 (review 1 blocker 2): hash-independent.
    # See `tj9e_support.money_that_leaked`.
    assert fx.money_that_leaked(body) == [], fx.money_that_leaked(body)


def test_every_leaf_value_in_the_evidence_is_one_the_desk_can_name(tmp_path):
    """The fence on the VALUES, not just the keys (review 1 blocker 2).

    `test_the_evidence_package_carries_these_keys_and_no_others` is a key-set
    equality, and the reviewer proved it PASSES with `" pnl 3972.66 at 191.83"`
    appended to `exit_session` - a key set cannot see a number smuggled inside
    a value. So every leaf here is asserted EQUAL to the thing it came from:
    the note's five fields are the note's five fields, the two picklists are
    what their loaders return, the instructions are the module's, and the two
    ids are hex derived from all of it. There is nowhere left to put a price.

    Hand-counted: 5 note leaves, 2 vocabularies, 3 housekeeping keys.
    """
    import exit_reasons
    import trader_state_tags
    from ai_jobs import exit_note_fields

    store, _trade_id = fx.swing_with_money(tmp_path)
    notes = exit_note_fields.notes_waiting(store, fx.REVIEWED)
    assert len(notes) == 1, notes
    note = notes[0]

    evidence = exit_note_fields.build_evidence(note)

    assert evidence["note"] == {
        "note_id": note["note_id"],
        "text": note["text"],
        "symbol": note["symbol"],
        "side": note["side"],
        "exit_session": note["exit_session"],
    }
    assert evidence["instructions"] == exit_note_fields.INSTRUCTIONS
    why = exit_reasons.load_vocabulary()
    felt = trader_state_tags.load_vocabulary()
    assert evidence["vocabularies"] == {
        "why": {
            "vocabulary_id": why["vocabulary_id"],
            "vocab_version": why["vocab_version"],
            "entries": [dict(entry) for entry in why["entries"]],
        },
        "felt": {
            "vocabulary_id": felt["vocabulary_id"],
            "vocab_version": felt["vocab_version"],
            "max_codes": exit_note_fields.MAX_FELT,
            "entries": [dict(entry) for entry in felt["entries"]],
        },
        "watching": {"max_quotes": exit_note_fields.MAX_WATCHING},
    }
    assert re.fullmatch(r"[0-9a-f]{64}", str(evidence["evidence_hash"]))
    assert str(evidence["package_id"]) == (
        f"exit-note-fields:{str(evidence['evidence_hash'])[:16]}"
    )
    assert set(evidence) == {
        "package_id", "evidence_hash", "instructions", "note", "vocabularies"
    }


def test_the_value_fence_catches_money_smuggled_into_a_value(tmp_path, monkeypatch):
    """The reviewer's own injection, as a standing test.

    `" pnl 3972.66 at 191.83"` appended to `exit_session` passes a key-set
    equality. It must not pass this one - and it must not pass
    `money_that_leaked` on the wire either.
    """
    from ai_jobs import exit_note_fields

    store, _trade_id = fx.swing_with_money(tmp_path)
    note = dict(exit_note_fields.notes_waiting(store, fx.REVIEWED)[0])
    note["exit_session"] = f"{note['exit_session']} pnl 97795.54 at 191.83"

    evidence = exit_note_fields.build_evidence(note)

    assert evidence["note"]["exit_session"] != fx.REVIEWED, "the injection vanished"
    assert fx.money_that_leaked(evidence) == [
        str(fx.MONEY_ENTRY_PRICE),
        str(fx.MONEY_NET_PNL),
    ], fx.money_that_leaked(evidence)


def test_the_exit_read_never_opens_the_shared_store_itself(tmp_path, monkeypatch):
    """Review 1 blocker 1, as a standing test.

    `journal_feed._store()` caches a module-global store for the life of the
    process, so whoever calls it FIRST decides which store the whole run uses -
    and runs its `initialize_schema()` migration on that caller's thread. Three
    Day-Review test files stub `DayReviewService._trades`; the exit-note read
    then cached a FAKE store and every later `JournalPanel` in the pytest
    process died on `db_path`, 32 errors that are green on base.

    Hand-counted: 0 calls to `_store` when the trades read returns nothing, the
    two keys still absent from a payload with no rows, and the payload SAYS it
    did not look.
    """
    from ui.services import journal_feed
    from ui.services.day_review_service import DayReviewService

    calls: list[str] = []
    monkeypatch.setattr(
        journal_feed, "_store", lambda: calls.append("store") or (_ for _ in ()).throw(
            AssertionError("the exit-note read opened the shared store")
        )
    )
    service = DayReviewService()
    monkeypatch.setattr(service, "_trades", lambda *a, **k: [])
    monkeypatch.setattr(service, "_read_recap", lambda *a, **k: object())

    payload = service.read_day(fx.REVIEWED, now=datetime(2026, 9, 14, 9, 0))

    assert calls == [], calls
    assert payload["trades"] == []
    assert "exit notes were not read" in str(payload.get("error") or ""), payload.get("error")


def test_the_report_card_counts_the_notes_this_payload_opened(tmp_path, monkeypatch):
    """Review 1 blocker 3: the line is WIRED, through `read_day`.

    `day_report_card.build` is the only production caller of `process_line` and
    never passed `exit_notes=`, so the live page said "nobody opened the exit
    notes" on the payload that had just opened every one of them. Tested here
    through `read_day` rather than by calling `process_line` - which is how the
    hole got through the first time.

    Hand-counted: 6 trades, 5 with an exit, 2 explained.
    """
    import trade_mentor_trade_check as check
    from ui.services import journal_feed
    from ui.services.day_review_service import DayReviewService

    store, ids = fx.ready_store(tmp_path)
    monkeypatch.setattr(journal_feed, "_store", lambda: store)
    for symbol in (fx.SWING, fx.DAY_TRADE):
        check.save_exit_note(
            store, ids[symbol], fx.EXIT_NOTE, exit_session=fx.REVIEWED,
            now=datetime.fromisoformat("2026-09-14T09:05:00-04:00"),
        )

    service = DayReviewService()
    monkeypatch.setattr(service, "_read_recap", lambda *a, **k: object())
    payload = service.read_day(fx.REVIEWED, now=datetime(2026, 9, 14, 9, 0))

    line = [
        row for row in (payload["report_card"] or {}).get("lines") or ()
        if str(row.get("key")) == "process"
    ]
    assert len(line) == 1, payload["report_card"]
    assert line[0]["exits_explained"] == 2, line[0]
    assert line[0]["exits_n"] == fx.EXIT_BOXES_EXPECTED, line[0]
    assert "exits explained 2 of 5" in str(line[0]["text"]).lower(), line[0]["text"]
    assert "nobody opened the exit notes" not in str(line[0]["text"]).lower()


def test_an_unexplained_exit_keeps_the_check_owed_on_a_later_slot(tmp_path, monkeypatch):
    """Review 1 blocker 4, through the REAL ride seam.

    A swing closed in the reviewed session with every entry field answered is
    not UNLABELLED, and it still has an exit nobody has explained. Before this
    the desk decided nothing was owed and returned before `set_trade_check`, so
    a trader who was away at 09:00 was never asked about that exit at all.

    Hand-counted: unlabelled 0, unexplained exits 1, owed TRUE; after the note
    is written, unexplained exits 0 and owed FALSE.
    """
    import trade_mentor_trade_check as check
    from ui.services.trade_mentor_service import TradeMentorService

    store, trade_id = fx.swing_only(tmp_path)
    service = TradeMentorService()

    assert service.unlabelled_trades(fx.REVIEWED, store=store) == 0
    assert service.unexplained_exits(fx.REVIEWED, store=store) == 1

    check.save_exit_note(
        store, trade_id, fx.EXIT_NOTE, exit_session=fx.REVIEWED,
        now=datetime.fromisoformat("2026-09-14T09:05:00-04:00"),
    )
    assert service.unexplained_exits(fx.REVIEWED, store=store) == 0


def test_the_ride_asks_again_while_an_exit_is_unexplained(tmp_path, monkeypatch):
    """The same blocker at the seam that decides it: `_trade_check_is_owed`.

    Driven through the real `MainWindow` method with the two counts stubbed at
    the SERVICE, so what is under test is the predicate and not the journal.
    """
    from ui.app import MainWindow

    class _Slot:
        scheduled_at = datetime(2026, 9, 14, 11, 0)

    class _Service:
        def __init__(self, unlabelled, exits):
            self._unlabelled = unlabelled
            self._exits = exits

        def unlabelled_trades(self, *_a, **_k):
            return self._unlabelled

        def unexplained_exits(self, *_a, **_k):
            return self._exits

    import trade_mentor_trade_check as check

    window = MainWindow.__new__(MainWindow)
    window.trade_mentor_service = _Service(0, 1)
    assert MainWindow._trade_check_is_owed(window, check, _Slot()) is True

    window.trade_mentor_service = _Service(1, 0)
    assert MainWindow._trade_check_is_owed(window, check, _Slot()) is True


def test_the_exit_draft_question_is_fed_and_can_fire(tmp_path, monkeypatch):
    """Review 1 advisory 2: the registry kind had no lane behind it.

    `mentor_questions.exit_draft_review` reads `state["exit_drafts"]` and
    nothing on the desk set that key, so lead decision 7's budget clause could
    never fire. Hand-counted: 1 waiting draft -> 1 subject; the same draft once
    CONFIRMED -> 0.
    """
    import mentor_questions
    import trade_mentor_trade_check as check
    from ai_jobs import exit_note_fields
    from ui.app import MainWindow

    store, trade_id, root = _swing_with_a_draft(tmp_path)
    # The lane reads the desk's OWN pack folder; point it at this test's.
    real_read = exit_note_fields.read_latest
    monkeypatch.setattr(
        exit_note_fields,
        "read_latest",
        lambda session, root=None: real_read(session, root=root if root is not None else globals()["_PACKS"]),
    )
    globals()["_PACKS"] = root

    lane = MainWindow._mentor_exit_drafts(store, fx.REVIEWED)
    assert [row["trade_id"] for row in lane] == [trade_id], lane
    assert mentor_questions._trigger_exit_draft_review({"exit_drafts": lane})

    check.confirm_exit_fields(
        store,
        trade_id,
        exit_note_fields.draft_for(trade_id, fx.REVIEWED, root=root),
        now=datetime.fromisoformat("2026-09-14T09:30:00-04:00"),
    )
    assert MainWindow._mentor_exit_drafts(store, fx.REVIEWED) == []


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
    """One swing with an ENTRY GAP, one exit note, and one night's draft.

    The entry is deliberately NOT answered. Since review 1 blocker 5 an exit
    the trader has explained is ANSWERED - it is not asked again - so a row
    whose entry is complete AND whose exit is explained is not on the card at
    all. A draft therefore only ever appears beside a row that is still listed
    for some other reason, which is the entry gap here.
    """
    import exit_reasons
    import trade_mentor_trade_check as check
    import trader_state_tags
    from ai_jobs import exit_note_fields

    store, _ids = fx.ready_store(tmp_path)
    trade_id = str(
        [row for row in store.list_trades(trade_date=fx.REVIEWED)
         if str(row.get("symbol")) == fx.DAY_TRADE][0]["trade_id"]
    )
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


def test_an_answered_exit_is_shown_back_and_never_asked_again(tmp_path):
    """Review 1 blocker 5. Hand-counted: 1 note, 0 boxes, 1 read-only line.

    The trader wrote the words an hour ago. The row is still on the card for
    its four entry gaps, so the exit shows what they WROTE and asks nothing -
    before this, the box came back empty, greyed Save, and could only be
    cleared by retyping the same sentence into a second `EXIT_NOTE_RAW` row.
    """
    import trade_mentor_trade_check as check

    store, trade_id, root = _swing_with_a_draft(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)
    row = [item for item in task.trades if item.trade_id == trade_id]
    assert len(row) == 1, [(q.symbol, q.missing) for q in task.trades]
    assert row[0].exit_answered is True, row[0]
    assert row[0].exit_note == fx.EXIT_NOTE, row[0]

    card = _card(tmp_path)
    card.set_trade_check(task, store=store, drafts_root=root)
    assert card.exit_note_box(trade_id) is None, "the answered exit was asked again"
    assert fx.EXIT_NOTE in card.exit_prompt_text(trade_id)
    assert card._exit_is_open(trade_id) is False


def test_an_explained_exit_with_no_entry_gap_leaves_the_card(tmp_path):
    """The other half of blocker 5: nothing open, nothing asked.

    Hand-counted: the swing has 0 entry gaps and 1 exit; before the note it is
    ONE row, after the note it is NONE - and `unexplained_exit_count` moves
    from 1 to 0 with it.
    """
    import trade_mentor_trade_check as check

    store, trade_id = fx.swing_only(tmp_path)
    assert check.unexplained_exit_count(store, fx.REVIEWED) == 1
    assert len(check.build_task(store, fx.SESSION_TODAY).trades) == 1

    check.save_exit_note(
        store, trade_id, fx.EXIT_NOTE, exit_session=fx.REVIEWED,
        now=datetime.fromisoformat("2026-09-14T09:05:00-04:00"),
    )

    assert check.unexplained_exit_count(store, fx.REVIEWED) == 0
    assert check.build_task(store, fx.SESSION_TODAY).trades == ()


def test_a_waiting_draft_never_greys_save(tmp_path):
    """A waiting draft is something to LOOK at, not a field to fill.

    Hand-counted: ONE row with four entry gaps and an answered exit. Grey while
    a field is open, green the moment the last one is answered - with the draft
    waiting the whole time. A reading nobody clicked must never be able to hold
    the morning hostage.
    """
    import trade_mentor_trade_check as check

    store, trade_id, root = _swing_with_a_draft(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)

    card = _card(tmp_path)
    card.set_trade_check(task, store=store, drafts_root=root)
    assert card.exit_draft_line(trade_id).strip(), "no draft line was shown"
    assert card.save_answers_button.isEnabled() is False, "nothing answered yet"

    for name, (combo, _text) in card._answer_inputs[trade_id].items():
        combo.setCurrentIndex(combo.findData(check.ANSWER_NOT_REMEMBERED))
    for other, fields in card._answer_inputs.items():
        if other == trade_id:
            continue
        for combo, _text in fields.values():
            combo.setCurrentIndex(combo.findData(check.ANSWER_NOT_REMEMBERED))
        box = card.exit_note_box(other)
        if box is not None:
            card.set_exit_answer_state(other, check.ANSWER_NOT_REMEMBERED)
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
