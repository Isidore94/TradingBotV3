r"""TJ-9E - the card tells an EXIT from an ENTRY. The six row shapes.

Trader, 2026-09-21: *"trade mentor should be able to differentiate between trade
entrys and exits ... trade entrys are good the way they are"*.

RED FOR (per test, measured on `claude/tj9e-exit-notes` at `05988440`):

* every test that names `exit_session`, `EVENT_EXIT_NOTE_RAW`, `save_exit_note`
  or `exit_notes` fails with `AttributeError` - `scripts/trade_mentor_trade_check.py`
  has none of those names;
* `test_a_partly_closed_trade_is_listed_on_the_card` fails on a real assertion
  (the trade is absent), because `_SESSION_STATUSES` spells the status
  `PARTIALLY_CLOSED` (`scripts/trade_mentor_trade_check.py:108`) and the live
  journal writes `CLOSED_PARTIAL` (`scripts/journal_store.py:1967`);
* `test_an_entry_only_row_is_byte_for_byte_what_it_is_today` is a STATED GUARD -
  it passes today and exists so the builder cannot change an entry.

PREMISE PINNED HERE. The packet believed `list_trades(trade_date=...)` keys on
the trade's OWN (open) date, so a swing closed yesterday "is not on this
morning's card at all". That is REFUTED: `scripts/journal_store.py:2134` matches
``(substr(opened_at,1,10) = ? OR substr(closed_at,1,10) = ? OR trade_date = ?)``
and `rebuild_trades` writes ``trade_date = closed_at or opened_at``
(`scripts/journal_store.py:1995`), so the swing IS returned. It never reaches
the card because `questions_for_session` `continue`s on an empty `missing`
(`scripts/trade_mentor_trade_check.py:564-565`). The packet's CONCLUSION stands
and its mechanism does not, so these tests are written to the code's truth: the
exit row must survive an EMPTY `missing`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj9e_support as fx  # noqa: E402


def _by_symbol(task):
    return {question.symbol: question for question in task.trades}


# ---------------------------------------------------------------------------
# the premise, stated as a test so a later reader cannot re-believe it
# ---------------------------------------------------------------------------
def test_the_store_already_returns_a_swing_closed_in_the_reviewed_session(tmp_path):
    """STATED GUARD (green today). `list_trades(trade_date=...)` is not the gap.

    Hand-counted: SWNG opened on 2026-09-04 and closed on 2026-09-11, so it is
    ONE of the six rows `list_trades(trade_date="2026-09-11")` returns. What
    drops it is the card, not the store.
    """
    store, _ids = fx.ready_store(tmp_path)

    symbols = {str(row.get("symbol") or "") for row in store.list_trades(trade_date=fx.REVIEWED)}
    assert fx.SWING in symbols, symbols
    assert len(symbols) == fx.ROWS_EXPECTED_AFTER_THE_FIX, sorted(symbols)


def test_the_live_partial_status_is_the_one_the_assembler_writes(tmp_path):
    """STATED GUARD (green today), and the proof the Mentor's spelling is wrong.

    The assembler stamps `CLOSED_PARTIAL`; nothing anywhere writes
    `PARTIALLY_CLOSED`. Asserted on the ASSEMBLED row so the test follows the
    assembler if it is ever renamed.
    """
    store, ids = fx.ready_store(tmp_path)

    part = store.get_trade(ids[fx.PARTLY_CLOSED])
    assert str(part.get("status") or "") == fx.LIVE_PARTIAL_STATUS, part.get("status")


# ---------------------------------------------------------------------------
# the six row shapes
# ---------------------------------------------------------------------------
def test_a_partly_closed_trade_is_listed_on_the_card(tmp_path):
    """THE STATUS DEFECT. Hand-counted: PART is 1 of the 6 rows, and today it is 0.

    Seven live trades are `CLOSED_PARTIAL` and every one of them is invisible to
    the 09:00 check, so the trade the trader half-exited is the one they are
    never asked about.
    """
    import trade_mentor_trade_check as check

    store, ids = fx.ready_store(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)

    rows = _by_symbol(task)
    assert fx.PARTLY_CLOSED in rows, sorted(rows)
    assert rows[fx.PARTLY_CLOSED].trade_id == ids[fx.PARTLY_CLOSED]


def test_every_trade_of_the_reviewed_session_gets_a_row_entry_gaps_or_not(tmp_path):
    """Hand-counted: 6 rows - OPNX, DAYT, SWNG, SCLO, PART, DTON.

    Today's code lists 4 (the table in `tj9e_support`): PART is dropped by the
    status spelling and SWNG by its empty `missing`. Both have to come back, and
    the count is asserted so "list everything" is not a way to pass either.
    """
    import trade_mentor_trade_check as check

    store, _ids = fx.ready_store(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)

    assert task.journal_ready is True
    assert task.reviewed_session == fx.REVIEWED
    assert len(task.trades) == fx.ROWS_EXPECTED_AFTER_THE_FIX, [
        (q.symbol, q.missing) for q in task.trades
    ]


def test_an_entry_only_row_is_byte_for_byte_what_it_is_today(tmp_path):
    """STATED GUARD. *"trade entrys are good the way they are"*.

    OPNX opened in the reviewed session and never exited. Hand-counted: all four
    material fields missing, in `MATERIAL_FIELDS` order, and NO exit asked. The
    four fields are read from the module, never spelled here.
    """
    import trade_mentor_trade_check as check

    store, _ids = fx.ready_store(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)

    row = _by_symbol(task)[fx.ENTRY_ONLY]
    assert row.missing == check.MATERIAL_FIELDS
    assert str(getattr(row, "exit_session", "")) == "", (
        "a trade with no closing fill was asked about an exit"
    )


def test_a_swing_closed_in_the_reviewed_session_is_an_exit_only_row(tmp_path):
    """Hand-counted: SWNG has 0 entry gaps and 1 exit, so the row is exit ONLY.

    Its entry was asked the morning after it opened and answered in full. The
    row must survive that - which is precisely what `questions_for_session`'s
    `if not gaps: continue` prevents today.
    """
    import trade_mentor_trade_check as check

    store, _ids = fx.ready_store(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)

    row = _by_symbol(task)[fx.SWING]
    assert row.missing == (), row.missing
    assert str(row.exit_session) == fx.REVIEWED


def test_a_day_trade_carries_the_four_entry_fields_and_the_exit_box_on_one_row(tmp_path):
    """Hand-counted: DAYT is ONE row with 4 missing fields AND 1 exit.

    Two rows for one trade would be two Save gates on the same trade, which is
    the thing `build_task`'s `seen` set already refuses for the session overlap.
    """
    import trade_mentor_trade_check as check

    store, ids = fx.ready_store(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)

    rows = [question for question in task.trades if question.symbol == fx.DAY_TRADE]
    assert len(rows) == 1, rows
    assert rows[0].trade_id == ids[fx.DAY_TRADE]
    assert rows[0].missing == check.MATERIAL_FIELDS
    assert str(rows[0].exit_session) == fx.REVIEWED


def test_a_scale_out_is_asked_once_per_trade_per_session(tmp_path):
    """Hand-counted: SCLO has TWO closing legs on 2026-09-11 and ONE exit box.

    Live, 70 trades have more than one closing leg. One question per FILL would
    make the forced morning a form, which is the failure mode the 09:00 check
    was already rewritten once to avoid.
    """
    import trade_mentor_trade_check as check

    store, ids = fx.ready_store(tmp_path)

    legs = [
        leg for leg in store.list_trade_legs(ids[fx.SCALE_OUT])
        if str(leg.get("role") or "").upper() == "CLOSE"
    ]
    assert len(legs) == 2, legs  # the fixture really does hold two closing fills

    task = check.build_task(store, fx.SESSION_TODAY)
    rows = [question for question in task.trades if question.symbol == fx.SCALE_OUT]
    assert len(rows) == 1, rows
    assert str(rows[0].exit_session) == fx.REVIEWED


def test_exactly_five_of_the_six_rows_are_asked_about_an_exit(tmp_path):
    """Hand-counted: 5 exit boxes (DAYT, SWNG, SCLO, PART, DTON), 1 without (OPNX).

    The whole count in one assertion, so a fix that produces an exit row for a
    trade with no closing fill fails here rather than being noticed live.
    """
    import trade_mentor_trade_check as check

    store, _ids = fx.ready_store(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)

    asked = {
        question.symbol for question in task.trades if str(question.exit_session or "")
    }
    assert asked == set(fx.TRADES_WITH_AN_EXIT), sorted(asked)
    assert len(asked) == fx.EXIT_BOXES_EXPECTED


# ---------------------------------------------------------------------------
# provenance - a broker file is blind to time
# ---------------------------------------------------------------------------
def test_a_date_only_exit_is_asked_and_its_note_is_never_same_session(tmp_path):
    """DTON's closing fill is stamped at MIDNIGHT market-local.

    Hand-counted: 1 live trade carries a date-only closing leg today (2 of 616
    executions are date-only, against the packet's believed 0), so this is a
    reachable case and not a hypothetical. `journal_trade_shape.is_date_only`
    is True for it, and "there is no time to be before", so a note typed on the
    fill's own date may never claim `same_session`.
    """
    import journal_trade_shape as shape
    import trade_mentor_trade_check as check
    import trade_origin
    from datetime import datetime

    store, ids = fx.ready_store(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)

    row = _by_symbol(task)[fx.DATE_ONLY_EXIT]
    assert str(row.exit_session) == fx.REVIEWED

    legs = [
        leg for leg in store.list_trade_legs(ids[fx.DATE_ONLY_EXIT])
        if str(leg.get("role") or "").upper() == "CLOSE"
    ]
    assert len(legs) == 1
    assert shape.is_date_only(shape._coerce_datetime(legs[0]["timestamp"])) is True

    # Typed on the exit's OWN date, which is the only moment that could make a
    # naive rule say `same_session`.
    written = check.save_exit_note(
        store,
        ids[fx.DATE_ONLY_EXIT],
        fx.EXIT_NOTE,
        exit_session=fx.REVIEWED,
        now=datetime.fromisoformat(f"{fx.REVIEWED}T11:00:00-04:00"),
    )
    payload = written["payload"]
    assert payload["label_provenance"] != trade_origin.SAME_SESSION, payload
    assert str(payload.get("label_provenance_reason") or "").strip(), payload


def test_an_exit_note_written_on_the_exits_own_session_is_a_same_session_note(tmp_path):
    """The EXIT's session is the ruler, not the trade's first fill.

    SWNG opened 2026-09-04 and exited 2026-09-11 with a real clock time. A note
    typed that afternoon is a same-session note ABOUT THE EXIT. Reusing
    `_answer_provenance` unchanged answers `recalled_after` here, because
    `trade_origin.trade_session` is the session of the FIRST FILL
    (`scripts/trade_origin.py:191-197`) - correct for an entry field, wrong for
    an exit. Live, 116 of 180 closed trades exit on a different date from their
    open, so this is the majority case.

    LEAD DECISION PINNED HERE (flagged in the handoff): an exit note's
    provenance is decided against the EXIT's session.
    """
    import trade_mentor_trade_check as check
    import trade_origin
    from datetime import datetime

    store, ids = fx.ready_store(tmp_path)

    written = check.save_exit_note(
        store,
        ids[fx.SWING],
        fx.EXIT_NOTE,
        exit_session=fx.REVIEWED,
        now=datetime.fromisoformat(f"{fx.REVIEWED}T13:40:00-04:00"),
    )
    payload = written["payload"]
    assert payload["label_provenance"] == trade_origin.SAME_SESSION, payload
    assert payload["exit_session"] == fx.REVIEWED
    assert payload["written_after_the_session"] is False, payload


def test_a_note_typed_the_next_morning_is_written_after_the_session(tmp_path):
    """The Monday card, about Friday's exit. `written_after_the_session` is
    COMPUTED and never backdated, and `label_provenance` is `recalled_after`."""
    import trade_mentor_trade_check as check
    import trade_origin
    from datetime import datetime

    store, ids = fx.ready_store(tmp_path)

    written = check.save_exit_note(
        store,
        ids[fx.SWING],
        fx.EXIT_NOTE,
        exit_session=fx.REVIEWED,
        now=datetime.fromisoformat("2026-09-14T09:05:00-04:00"),
    )
    payload = written["payload"]
    assert payload["label_provenance"] == trade_origin.RECALLED_AFTER, payload
    assert payload["written_after_the_session"] is True, payload
