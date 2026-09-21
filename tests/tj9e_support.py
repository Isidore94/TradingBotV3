r"""Hand-built fixtures for the TJ-9E (the Mentor tells an EXIT from an ENTRY)
red tests. NOT a test module.

Nothing here is produced by the code under test. `scripts/ai_jobs/exit_note_fields.py`,
`scripts/exit_reasons.py` and the exit box on the card do not exist on this
branch (verified 2026-09-21 at `05988440`: no `exit_note`, no `exit_reasons` and
no `EXIT_NOTE_RAW` anywhere under `scripts/`). Every trade below is assembled by
the REAL import path (`manual_execution_from_fields` -> `upsert_executions` ->
`rebuild_trades`), and every expected number in a test is hand-counted in that
test's own docstring.

THE LIVE SHAPE THIS MODELS (read-only counts, 2026-09-21, stdlib sqlite3
``mode=ro`` over ``C:\TradingBotData\data\runtime\trade_journal.sqlite3``)
-------------------------------------------------------------------------
=========================================  =====
trades                                       216
  CLOSED                                     180
  OPEN                                        29
  CLOSED_PARTIAL                               7
closed trades whose CLOSE date != OPEN date  116
trades with more than one closing leg         70
trades with closing legs on TWO dates         13
executions with a date-only (midnight) stamp   2  (of 616)
`opportunity_events` RECALLED / RECALLED_RAW  1 / 0
=========================================  =====

So: an exit in a session other than the open is the MAJORITY case (116 of 180
closed trades), a scale-out is ordinary (70), a scale-out spanning two sessions
is real (13) and a date-only exit exists (1 trade, and the packet believed 0).
The honest first state of every exit-note store is EMPTY - `RECALLED_RAW` is 0
live and `EXIT_NOTE_RAW` is a name the code does not yet know.

THE SIX TRADES THIS MODULE BUILDS, HAND-COUNTED
-----------------------------------------------
Reviewed session is Friday :data:`REVIEWED` (2026-09-11); the card is Monday
:data:`SESSION_TODAY` (2026-09-14). `PRIOR` (2026-09-04) is an earlier session.

===========  ==========================  ==========  ============  ==========
symbol       fills                       status      entry gaps    exits in
                                                                   REVIEWED
===========  ==========================  ==========  ============  ==========
OPNX         BUY 100 on REVIEWED         OPEN        4             0
DAYT         BUY+SELL 100 on REVIEWED    CLOSED      4             1
SWNG         BUY on PRIOR, SELL on       CLOSED      0 (answered)  1
             REVIEWED
SCLO         BUY 100, SELL 40, SELL 60   CLOSED      4             2 legs,
             all on REVIEWED                                       ONE ask
PART         BUY 100, SELL 40 on         CLOSED_     4             1
             REVIEWED                    PARTIAL
DTON         BUY on PRIOR (clock time),  CLOSED      4             1
             SELL on REVIEWED at
             T00:00:00 (date-only)
===========  ==========================  ==========  ============  ==========

Hand-counted totals for the reviewed session:

* trades with an exit in REVIEWED: **5** (DAYT, SWNG, SCLO, PART, DTON)
* exit boxes on the card: **5** - SCLO's two closing legs are ONE ask
* entry-only rows (an entry gap and no exit): **1** (OPNX)
* rows with BOTH (a day trade): **1** (DAYT) - plus SCLO, PART and DTON, which
  also opened in REVIEWED, so **4** rows carry both halves
* exit-only rows (opened earlier, nothing left to ask about the entry): **1**
  (SWNG)
* what TODAY's code lists: **4** (OPNX, DAYT, SCLO, DTON). PART is dropped by
  `_SESSION_STATUSES`' `PARTIALLY_CLOSED` spelling; SWNG is dropped because
  `questions_for_session` `continue`s on an empty `missing`.
* what the fix must list: **6**.

WHAT IS MODELLED AS IT REALLY IS
--------------------------------
* A partly closed trade's status is the string the LIVE journal writes,
  ``CLOSED_PARTIAL`` (`scripts/journal_store.py:1967`), never the
  ``PARTIALLY_CLOSED`` the Trade Mentor spells.
* A date-only exit is stamped at midnight market-local, which is what the
  statement importer writes and what `journal_trade_shape.is_date_only` reads.
* A note text carries a MULTI-BYTE character, so a span expressed in characters
  and a span expressed in bytes give different answers.
* A raw note row is an append-only `opportunity_events` row; a superseding note
  APPENDS and leaves the first row byte-identical.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
for _extra in (SCRIPTS_DIR, ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

#: A plain Monday. Its previous exchange session is Friday 2026-09-11.
SESSION_TODAY = date(2026, 9, 14)
REVIEWED = "2026-09-11"
#: An earlier regular session - where the swing and the date-only trade opened.
PRIOR = "2026-09-04"
#: Never a real account number.
ACCOUNT = "TJ9E-TEST-ACCOUNT"

#: The six symbols, by the shape each one is here to model.
ENTRY_ONLY = "OPNX"
DAY_TRADE = "DAYT"
SWING = "SWNG"
SCALE_OUT = "SCLO"
PARTLY_CLOSED = "PART"
DATE_ONLY_EXIT = "DTON"

#: Hand-counted from the table above.
TRADES_WITH_AN_EXIT = (DAY_TRADE, SWING, SCALE_OUT, PARTLY_CLOSED, DATE_ONLY_EXIT)
EXIT_BOXES_EXPECTED = 5
ROWS_EXPECTED_AFTER_THE_FIX = 6
ROWS_LISTED_BY_TODAYS_CODE = 4

#: The status string the live journal writes for a half-exited position. Spelled
#: out ONCE here so a test says which of the two spellings it means; the code
#: under test must not spell it a third time.
LIVE_PARTIAL_STATUS = "CLOSED_PARTIAL"

#: The trader's own words about an exit. Deliberately:
#:  * it holds NO number, no price, no P&L and no R, so a payload that leaked
#:    one is obvious;
#:  * it holds a MULTI-BYTE character (the en dash), so a span counted in bytes
#:    lands in the wrong place;
#:  * it names a reason, a feeling and a thing watched, so all three fields are
#:    reachable from one note.
EXIT_NOTE = (
    "I took it off when the 50 day cracked \u2013 felt rushed after the open and "
    "I was watching the volume dry up on the retest."
)

#: Exact substrings of :data:`EXIT_NOTE`, each one a real quote a draft may
#: cite. Their offsets are COMPUTED from the note (never typed), because a
#: hand-typed offset is a second copy of the note.
QUOTE_WHY = "the 50 day cracked"
QUOTE_FELT = "felt rushed"
QUOTE_WATCHING = "the volume dry up on the retest"

#: A second note on the same exit - the supersede case.
SECOND_EXIT_NOTE = "On reflection I exited because I needed the capital for the open."


def span_of(phrase: str, text: str = EXIT_NOTE) -> list[int]:
    """The CHARACTER span of `phrase` inside `text`, as the reply must carry it.

    Computed, never typed. `str.index` counts characters, and the note holds a
    multi-byte character, so a builder that measures the note in bytes produces
    a different pair and the grounding check rejects it.
    """
    start = str(text).index(str(phrase))
    return [start, start + len(str(phrase))]


# ---------------------------------------------------------------------------
# the store and the six trades
# ---------------------------------------------------------------------------
def new_store(tmp_path: Path):
    """A real journal database, empty, under `tmp_path`."""
    from journal_store import JournalStore

    return JournalStore(Path(tmp_path) / "journal.sqlite3")


def _execution(
    execution_id: str,
    *,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    timestamp: str,
) -> Any:
    from journal_importers import manual_execution_from_fields

    return manual_execution_from_fields(
        {
            "broker": "MANUAL",
            "account_number": ACCOUNT,
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


def build_six_trades(store) -> dict[str, str]:
    """Assemble the six trades of this module's table. Returns `{symbol: trade_id}`.

    ONE `rebuild_trades` for all of them, the way an import does it, so the
    statuses and legs are the real assembler's and not this module's opinion.
    """
    rows = [
        # OPNX - an entry with no exit at all.
        _execution(f"{ENTRY_ONLY}-1", symbol=ENTRY_ONLY, side="BUY", qty=100,
                   price=10.0, timestamp=f"{REVIEWED}T07:31:00"),
        # DAYT - opened and closed in the reviewed session.
        _execution(f"{DAY_TRADE}-1", symbol=DAY_TRADE, side="BUY", qty=100,
                   price=20.0, timestamp=f"{REVIEWED}T07:35:00"),
        _execution(f"{DAY_TRADE}-2", symbol=DAY_TRADE, side="SELL", qty=100,
                   price=21.0, timestamp=f"{REVIEWED}T09:05:00"),
        # SWNG - opened a week earlier, closed in the reviewed session.
        _execution(f"{SWING}-1", symbol=SWING, side="BUY", qty=100,
                   price=30.0, timestamp=f"{PRIOR}T07:40:00"),
        _execution(f"{SWING}-2", symbol=SWING, side="SELL", qty=100,
                   price=33.0, timestamp=f"{REVIEWED}T10:20:00"),
        # SCLO - one entry, TWO closing legs, both in the reviewed session.
        _execution(f"{SCALE_OUT}-1", symbol=SCALE_OUT, side="BUY", qty=100,
                   price=40.0, timestamp=f"{REVIEWED}T07:31:00"),
        _execution(f"{SCALE_OUT}-2", symbol=SCALE_OUT, side="SELL", qty=40,
                   price=41.0, timestamp=f"{REVIEWED}T09:05:00"),
        _execution(f"{SCALE_OUT}-3", symbol=SCALE_OUT, side="SELL", qty=60,
                   price=42.0, timestamp=f"{REVIEWED}T10:05:00"),
        # PART - half exited, so the live assembler stamps it CLOSED_PARTIAL.
        _execution(f"{PARTLY_CLOSED}-1", symbol=PARTLY_CLOSED, side="BUY", qty=100,
                   price=50.0, timestamp=f"{REVIEWED}T07:32:00"),
        _execution(f"{PARTLY_CLOSED}-2", symbol=PARTLY_CLOSED, side="SELL", qty=40,
                   price=51.0, timestamp=f"{REVIEWED}T11:00:00"),
        # DTON - a real entry stamp, and a DATE-ONLY exit at midnight.
        _execution(f"{DATE_ONLY_EXIT}-1", symbol=DATE_ONLY_EXIT, side="BUY", qty=100,
                   price=60.0, timestamp=f"{PRIOR}T08:15:00"),
        _execution(f"{DATE_ONLY_EXIT}-2", symbol=DATE_ONLY_EXIT, side="SELL", qty=100,
                   price=61.0, timestamp=f"{REVIEWED}T00:00:00"),
    ]
    store.upsert_executions(rows)
    store.rebuild_trades(refresh_tags=False)

    ids: dict[str, str] = {}
    for day in (PRIOR, REVIEWED):
        for trade in store.list_trades(trade_date=day):
            ids.setdefault(str(trade.get("symbol") or ""), str(trade["trade_id"]))
    missing = [
        name
        for name in (ENTRY_ONLY, DAY_TRADE, SWING, SCALE_OUT, PARTLY_CLOSED, DATE_ONLY_EXIT)
        if name not in ids
    ]
    assert not missing, f"these did not assemble into trades: {missing}"
    return ids


def answer_every_entry_field(store, trade_id: str) -> None:
    """Make a trade's four material fields ANSWERED, through the real writer.

    `not_remembered` is a complete answer, so `missing_fields` returns ``()``
    afterwards and TODAY's `questions_for_session` drops the trade entirely.
    That is exactly the state a swing closed yesterday is in, and it is why
    nothing on this morning's card ever asks about its exit.
    """
    import trade_mentor_trade_check as check

    check.save_answers(
        store,
        trade_id,
        {name: {"state": check.ANSWER_NOT_REMEMBERED} for name in check.MATERIAL_FIELDS},
    )


def mark_covered(store, day: str = REVIEWED, *, status: str = "COVERED") -> None:
    """The broker statement for `day` landed - what `_journal_ready` reads."""
    import journal_coverage

    journal_coverage.mark_range(
        store,
        broker="QUESTRADE",
        account_number=ACCOUNT,
        start=day,
        end=day,
        status=status,
        source="tj9e-test",
    )


def ready_store(tmp_path: Path) -> tuple[Any, dict[str, str]]:
    """The whole fixture in one call: a covered reviewed session and six trades.

    SWNG's entry fields are answered here, so it is the exit-ONLY row.
    """
    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    ids = build_six_trades(store)
    answer_every_entry_field(store, ids[SWING])
    return store, ids


def swing_only(tmp_path: Path) -> tuple[Any, str]:
    """A store holding ONE trade: the swing, entry answered, exit unexplained.

    The exit-ONLY row, alone on the card, so a Save-gate test can say the exit
    box is the WHOLE gate without six other boxes standing behind it.
    """
    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    store.upsert_executions(
        [
            _execution(f"{SWING}-1", symbol=SWING, side="BUY", qty=100,
                       price=30.0, timestamp=f"{PRIOR}T07:40:00"),
            _execution(f"{SWING}-2", symbol=SWING, side="SELL", qty=100,
                       price=33.0, timestamp=f"{REVIEWED}T10:20:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    trades = store.list_trades(trade_date=REVIEWED)
    assert len(trades) == 1, trades
    trade_id = str(trades[0]["trade_id"])
    answer_every_entry_field(store, trade_id)
    return store, trade_id


def day_trade_only(tmp_path: Path) -> tuple[Any, str]:
    """A store holding ONE trade: opened and closed in the reviewed session.

    The BOTH row, alone on the card: four entry fields and one exit box.
    """
    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    store.upsert_executions(
        [
            _execution(f"{DAY_TRADE}-1", symbol=DAY_TRADE, side="BUY", qty=100,
                       price=20.0, timestamp=f"{REVIEWED}T07:35:00"),
            _execution(f"{DAY_TRADE}-2", symbol=DAY_TRADE, side="SELL", qty=100,
                       price=21.0, timestamp=f"{REVIEWED}T09:05:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    trades = store.list_trades(trade_date=REVIEWED)
    assert len(trades) == 1, trades
    return store, str(trades[0]["trade_id"])


#: Prices and a P&L no JSON grammar hint, span offset or token count can collide
#: with. They exist so "this number is not in the request body" is a real
#: assertion rather than a coincidence about small integers.
#:
#: LEAD-GRANTED AMENDMENT 2026-09-21 (review 1 blocker 2). `MONEY_QUANTITY` was
#: `73`, and the guard that says no money reached the prompt searched the WHOLE
#: serialised body - which carries `evidence_hash`, a sha256 whose input
#: includes a random `note_id`. About half of all runs produced a digest
#: containing the characters `73`, so the packet's headline fence was pinned by
#: a test that failed 4 times in 8 while nothing leaked. Every value here now
#: carries a decimal point or is long enough that a hex collision is not the
#: likely explanation, and :func:`money_that_leaked` searches a copy of the body
#: with the long hex ids removed, matching the quantity on WORD BOUNDARIES.
MONEY_ENTRY_PRICE = 191.83
MONEY_EXIT_PRICE = 205.17
MONEY_QUANTITY = 7331
#: Hand-computed: (205.17 - 191.83) * 7331 = 97795.54, and the fixture charges
#: no commission or fees, so the trade's `net_pnl` is that number exactly.
MONEY_NET_PNL = 97795.54

#: A run of hex long enough to be an ID rather than a number the desk wrote: a
#: sha256 digest, the 16-character `package_id` suffix, or a 32-character event
#: id. Money never lives inside one, and a two-digit or four-digit string does,
#: often enough to break a guard half the time.
_HEX_ID = re.compile(r"[0-9a-fA-F]{16,}")


def wire_text(body: Any) -> str:
    """The request body as it goes on the wire, with the long hex IDS removed.

    Not a looser search: it removes only runs of 16+ hex characters, which are
    identifiers by construction. Everything the desk actually wrote - the
    note, the symbol, the side, every code, every number - survives.
    """
    return _HEX_ID.sub("<id>", json.dumps(body, default=str, ensure_ascii=False))


def money_that_leaked(body: Any) -> list[str]:
    """Every money value of :func:`swing_with_money` that reached the wire.

    ``[]`` is the only acceptable answer. The three prices carry a decimal
    point so they cannot hide inside anything else; the quantity is matched on
    WORD BOUNDARIES, so `7331` inside a longer number or an id is not a leak
    and `7331` standing alone is.
    """
    text = wire_text(body)
    found = [
        str(value)
        for value in (MONEY_ENTRY_PRICE, MONEY_EXIT_PRICE, MONEY_NET_PNL)
        if str(value) in text
    ]
    if re.search(rf"(?<![0-9a-fA-F.]){MONEY_QUANTITY}(?![0-9a-fA-F.])", text):
        found.append(str(MONEY_QUANTITY))
    return found


def swing_with_money(tmp_path: Path) -> tuple[Any, str]:
    """One closed swing whose numbers are DISTINCTIVE, with an exit note saved.

    Everything the night may never see is really sitting in the store the slot
    reads from: an entry price, an exit price, a quantity and a realised P&L.
    Hand-counted: 1 trade, 2 executions, 2 legs, 1 `EXIT_NOTE_RAW` row.
    """
    from datetime import datetime as _datetime

    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    store.upsert_executions(
        [
            _execution("MONEY-1", symbol=SWING, side="BUY", qty=MONEY_QUANTITY,
                       price=MONEY_ENTRY_PRICE, timestamp=f"{PRIOR}T07:40:00"),
            _execution("MONEY-2", symbol=SWING, side="SELL", qty=MONEY_QUANTITY,
                       price=MONEY_EXIT_PRICE, timestamp=f"{REVIEWED}T10:20:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    trades = store.list_trades(trade_date=REVIEWED)
    assert len(trades) == 1, trades
    trade_id = str(trades[0]["trade_id"])
    assert round(float(trades[0]["net_pnl"]), 2) == MONEY_NET_PNL, trades[0]["net_pnl"]
    check.save_exit_note(
        store,
        trade_id,
        EXIT_NOTE,
        exit_session=REVIEWED,
        now=_datetime.fromisoformat("2026-09-14T09:05:00-04:00"),
    )
    return store, trade_id


def notes_waiting_store(tmp_path: Path, count: int) -> tuple[Any, list[str]]:
    """`count` closed day trades on the reviewed session, each with ONE raw note.

    The notes are saved a minute apart in the order the trades were made, so
    "oldest first" is a real ordering and not the order of a dict.
    """
    from datetime import datetime as _datetime, timedelta as _timedelta

    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    rows = []
    for index in range(int(count)):
        symbol = f"SY{index:03d}"
        rows.append(_execution(f"{symbol}-1", symbol=symbol, side="BUY", qty=10,
                               price=10.0 + index, timestamp=f"{REVIEWED}T07:35:00"))
        rows.append(_execution(f"{symbol}-2", symbol=symbol, side="SELL", qty=10,
                               price=11.0 + index, timestamp=f"{REVIEWED}T09:05:00"))
    store.upsert_executions(rows)
    store.rebuild_trades(refresh_tags=False)

    by_symbol = {
        str(trade["symbol"]): str(trade["trade_id"])
        for trade in store.list_trades(trade_date=REVIEWED)
    }
    base = _datetime.fromisoformat("2026-09-14T09:00:00-04:00")
    ordered: list[str] = []
    for index in range(int(count)):
        trade_id = by_symbol[f"SY{index:03d}"]
        check.save_exit_note(
            store,
            trade_id,
            EXIT_NOTE,
            exit_session=REVIEWED,
            now=base + _timedelta(minutes=index),
        )
        ordered.append(trade_id)
    return store, ordered


def slot_at(session: date, hour: int):
    """The real scheduled slot for `hour` on `session`, or an error."""
    from trade_mentor_schedule import slots_for_session

    for slot in slots_for_session(session):
        if slot.scheduled_at.hour == hour:
            return slot
    raise AssertionError(f"no {hour:02d}:00 slot on {session}")


# ---------------------------------------------------------------------------
# the fake model - the only "model" any TJ-9E test ever sees
# ---------------------------------------------------------------------------
def fake_request(answer: Mapping[str, Any] | Sequence[Mapping[str, Any]], *, calls: list | None = None):
    """A `request=` injection that records its kwargs and answers.

    With a SEQUENCE it answers each call in turn (and repeats the last one when
    it runs out), which is how a night of several notes is driven.
    """
    replies = [dict(answer)] if isinstance(answer, Mapping) else [dict(row) for row in answer]

    def _request(**kwargs):
        if calls is not None:
            calls.append(dict(kwargs))
        index = min(len(calls or []) - 1 if calls is not None else 0, len(replies) - 1)
        return {"summary": dict(replies[max(index, 0)]), "model": "tj9e-fake-medium"}

    return _request


class _Response:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


def fake_post(reply: Mapping[str, Any], calls: list):
    """A fake local endpoint that answers `reply` and records the REQUEST BODY.

    The body is what the no-outcome test asserts on: the bytes the desk would
    have put on the wire, not a structure the slot handed itself.
    """

    def _post(url, **kwargs):
        calls.append(dict(kwargs))
        return _Response(
            {
                "id": "chatcmpl-exit-note",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": json.dumps(dict(reply))},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 400, "completion_tokens": 90, "total_tokens": 490},
            }
        )

    return _post


# ---------------------------------------------------------------------------
# replies, built from the note so a span always reproduces its quote
# ---------------------------------------------------------------------------
def value(phrase: str, code: str = "", *, text: str = EXIT_NOTE) -> dict[str, Any]:
    """One grounded value: a span that really does reproduce its quote."""
    row: dict[str, Any] = {"span": span_of(phrase, text), "quote": phrase}
    if code:
        row["code"] = code
    return row


def good_reply(
    why_code: str,
    felt_codes: Sequence[str],
    *,
    watching: Sequence[str] = (QUOTE_WATCHING,),
    text: str = EXIT_NOTE,
) -> dict[str, Any]:
    """A reply every bound of which holds. Hand-built, never round-tripped."""
    return {
        "fields": {
            "why": value(QUOTE_WHY, why_code, text=text),
            "felt": [value(QUOTE_FELT, code, text=text) for code in felt_codes],
            "watching": [value(phrase, text=text) for phrase in watching],
        }
    }


__all__ = [
    "ACCOUNT", "DATE_ONLY_EXIT", "DAY_TRADE", "ENTRY_ONLY", "EXIT_BOXES_EXPECTED",
    "EXIT_NOTE", "LIVE_PARTIAL_STATUS", "PARTLY_CLOSED", "PRIOR", "QUOTE_FELT",
    "QUOTE_WATCHING", "QUOTE_WHY", "REVIEWED", "ROWS_EXPECTED_AFTER_THE_FIX",
    "ROWS_LISTED_BY_TODAYS_CODE", "SCALE_OUT", "SECOND_EXIT_NOTE", "SESSION_TODAY",
    "SWING", "TRADES_WITH_AN_EXIT", "answer_every_entry_field", "build_six_trades",
    "fake_post", "fake_request", "good_reply", "mark_covered", "new_store",
    "ready_store", "slot_at", "span_of", "swing_only", "day_trade_only", "value",
    "MONEY_ENTRY_PRICE", "MONEY_EXIT_PRICE", "MONEY_NET_PNL", "MONEY_QUANTITY",
    "money_that_leaked", "notes_waiting_store", "swing_with_money", "wire_text",
]
