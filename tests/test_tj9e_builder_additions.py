r"""TJ-9E - the two things the tester did not pin, written by the BUILDER.

Two gaps, each with its own fail-before-fix proof recorded in the commit that
introduced it:

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
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj9e_support as fx  # noqa: E402

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
    second = fx.good_reply(
        exit_reasons.codes()[1],
        trader_state_tags.codes()[:1],
        watching=("the capital for the open",),
        text=fx.SECOND_EXIT_NOTE,
    )
    second["fields"]["why"] = fx.value(
        "I needed the capital", exit_reasons.codes()[1], text=fx.SECOND_EXIT_NOTE
    )
    second["fields"]["felt"] = []

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
