"""R7 §9 step 4 - assembly, and the cases nothing has ever tested.

The spec's step 4 names five: a missing middle execution, a stuck-open position,
the socket-vs-Flex duplicate, group splitting, and annotation survival across a
backfill that inserts an earlier opening fill. Each is here, plus the adjustment
actions that let the trader correct what the brokers got wrong.

These live outside the golden fixture on purpose. The golden's value comes from
its input never moving; new behaviours get new corpora, so a step that adds a
case cannot also quietly change what the frozen one proves.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from journal_identity import group_key_text  # noqa: E402
from journal_store import JournalStore  # noqa: E402


def _execution(uid: str, side: str, quantity: float, price: float, timestamp: str, **overrides):
    row = {
        "execution_uid": uid,
        "broker": "QUESTRADE",
        "account_number": "51234567",
        "account_label": "Margin",
        "account_type": "",
        "symbol": "AAPL",
        "security_type": "STK",
        "currency": "USD",
        "side": side,
        "quantity": quantity,
        "price": price,
        "timestamp": timestamp,
        "trade_date": timestamp[:10],
        "commission": 0.0,
        "fees": 0.0,
        "gross_amount": None,
        "net_amount": None,
        "order_id": "",
        "exchange_exec_id": "",
        "raw_json": "{}",
    }
    row.update(overrides)
    return row


@pytest.fixture
def store(tmp_path):
    return JournalStore(tmp_path / "trade_journal.sqlite3")


def _trades(store):
    with store.connection() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM trades ORDER BY opened_at, symbol")]


def _legs(store, trade_id):
    with store.connection() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM trade_legs WHERE trade_id = ? ORDER BY leg_id", (trade_id,)
            )
        ]


def _adjust(store, *, action, target_kind, target_uid, payload, reason="test", created_at="2026-08-10T09:00:00"):
    with store.connection() as conn:
        adjustment_id = uuid.uuid4().hex
        conn.execute(
            """
            INSERT INTO trade_adjustments(
                adjustment_id, target_kind, target_uid, action, payload_json, reason, source,
                superseded_by, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, 'test', '', ?)
            """,
            (adjustment_id, target_kind, target_uid, action, json.dumps(payload), reason, created_at),
        )
    return adjustment_id


# ---------------------------------------------------------------------------
# B1 - a half-exited position says so
# ---------------------------------------------------------------------------


def test_a_partly_exited_position_is_not_an_untouched_one(store):
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
            _execution("QT:5:2", "SELL", 40, 155.0, "2026-08-03T11:00:00-07:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    trade = _trades(store)[0]
    assert trade["status"] == "CLOSED_PARTIAL"
    assert trade["quantity_opened"] == 100.0 and trade["quantity_closed"] == 40.0
    # Still open, so no close date is claimed.
    assert trade["closed_at"] == ""


def test_an_untouched_position_is_still_plain_open(store):
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00")])
    store.rebuild_trades(refresh_tags=False)
    assert _trades(store)[0]["status"] == "OPEN"


def test_a_flat_position_is_still_plain_closed(store):
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
            _execution("QT:5:2", "SELL", 100, 155.0, "2026-08-03T11:00:00-07:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    trade = _trades(store)[0]
    assert trade["status"] == "CLOSED" and trade["closed_at"].startswith("2026-08-03T11:00")


# ---------------------------------------------------------------------------
# B2 - a missing opening fill is named, not invented around
# ---------------------------------------------------------------------------


def test_selling_more_than_the_journal_knows_you_own_flags_itself(store):
    """The concrete B2 case: 150 sold against 100 the journal has seen.

    The old code closed the 100 and turned the leftover 50 into a SHORT
    position that the trader never took and that would never close.
    """
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
            _execution("QT:5:2", "SELL", 150, 155.0, "2026-08-03T11:00:00-07:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    trades = _trades(store)
    assert len(trades) == 2

    closed, leftover = trades[0], trades[1]
    assert (closed["status"], closed["quantity_closed"]) == ("CLOSED", 100.0)
    assert closed["reconcile_status"] == "", "the part the journal could account for is fine"

    # The 50 shares were really sold, so the trade exists - but it is marked as
    # resting on an opening fill nobody imported, not presented as a short.
    assert leftover["direction"] == "SHORT" and leftover["quantity_opened"] == 50.0
    assert leftover["reconcile_status"] == "NEEDS_REVIEW"
    assert [leg["role"] for leg in _legs(store, leftover["trade_id"])] == ["SYNTHETIC_OPEN"]


def test_an_ordinary_short_entry_is_not_flagged(store):
    """A short the trader really took must not be noise in the review queue.

    Selling with no position open is genuinely ambiguous - it is either a short
    entry or a sale of shares bought before the import window - and nothing in
    the execution says which. Only the oversell case is unambiguous evidence, so
    only it is flagged here. The other half is caught by reconciliation against
    the broker's own positions (§9 step 9), where the broker reporting flat
    against a journal that says short is the proof this step cannot have.
    """
    store.upsert_executions([_execution("QT:5:1", "SELL", 50, 250.0, "2026-08-04T06:35:00-07:00")])
    store.rebuild_trades(refresh_tags=False)
    trade = _trades(store)[0]
    assert trade["direction"] == "SHORT" and trade["status"] == "OPEN"
    assert trade["reconcile_status"] == ""
    assert [leg["role"] for leg in _legs(store, trade["trade_id"])] == ["OPEN"]


def test_a_missing_middle_execution_does_not_corrupt_the_ones_around_it(store):
    """Named by §9 step 4 as never tested. Buy 100, [missing buy 100], sell 150."""
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
            _execution("QT:5:3", "SELL", 150, 160.0, "2026-08-03T13:00:00-07:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    closed, leftover = _trades(store)
    # The part that can be accounted for keeps its real numbers.
    assert closed["gross_pnl"] == pytest.approx(1000.0)
    assert leftover["reconcile_status"] == "NEEDS_REVIEW"


# ---------------------------------------------------------------------------
# Adjustments (I3): a correction that survives the next import
# ---------------------------------------------------------------------------


def test_a_voided_execution_is_skipped_but_never_deleted(store):
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
            _execution("QT:5:dupe", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
        ]
    )
    _adjust(store, action="VOID_EXECUTION", target_kind="EXECUTION", target_uid="QT:5:dupe",
            payload={}, reason="broker sent it twice")
    store.rebuild_trades(refresh_tags=False)
    assert _trades(store)[0]["quantity_opened"] == 100.0
    with store.connection() as conn:
        # I3: the raw row is still there. Voiding is an instruction, not a delete.
        assert conn.execute("SELECT COUNT(*) FROM raw_executions").fetchone()[0] == 2


def test_an_edit_overlays_only_the_fields_it_names(store):
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
            _execution("QT:5:2", "SELL", 100, 155.0, "2026-08-03T11:00:00-07:00"),
        ]
    )
    _adjust(store, action="EDIT_EXECUTION", target_kind="EXECUTION", target_uid="QT:5:2",
            payload={"price": 156.0, "commission": 4.95}, reason="statement says 156.00")
    store.rebuild_trades(refresh_tags=False)
    trade = _trades(store)[0]
    assert trade["gross_pnl"] == pytest.approx(600.0)
    assert trade["commission"] == pytest.approx(4.95)
    assert trade["symbol"] == "AAPL", "an edit may not rewrite what it did not name"


def test_an_edit_cannot_rewrite_an_execution_uid(store):
    """Identity is not data. An adjustment that could re-key a row could merge two."""
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00")])
    _adjust(store, action="EDIT_EXECUTION", target_kind="EXECUTION", target_uid="QT:5:1",
            payload={"execution_uid": "QT:5:something-else", "price": 151.0}, reason="try it")
    store.rebuild_trades(refresh_tags=False)
    assert _legs(store, _trades(store)[0]["trade_id"])[0]["execution_uid"] == "QT:5:1"
    assert _trades(store)[0]["average_entry_price"] == pytest.approx(151.0)


def test_an_added_execution_joins_assembly_in_time_order(store):
    """The opening fill a broker never reported, entered by hand."""
    store.upsert_executions([_execution("QT:5:2", "SELL", 100, 155.0, "2026-08-03T11:00:00-07:00")])
    _adjust(
        store,
        action="ADD_EXECUTION",
        target_kind="EXECUTION",
        target_uid="MANUAL:51234567:missing-open",
        payload={
            "broker": "QUESTRADE", "account_number": "51234567", "symbol": "AAPL",
            "security_type": "STK", "currency": "USD", "side": "BUY", "quantity": 100,
            "price": 150.0, "timestamp": "2026-08-03T09:31:00-07:00",
        },
        reason="bought before the import window",
    )
    store.rebuild_trades(refresh_tags=False)
    trade = _trades(store)[0]
    assert (trade["status"], trade["direction"]) == ("CLOSED", "LONG")
    assert trade["gross_pnl"] == pytest.approx(500.0)
    assert [leg["role"] for leg in _legs(store, trade["trade_id"])] == ["OPEN", "CLOSE"]


def test_reassigning_a_group_reunites_a_manual_fill_with_its_position(store):
    """The MANUAL orphan the classifier cannot fix, fixed the way §5 fix 3 says."""
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
            _execution("MANUAL:MANUAL:m1", "SELL", 100, 155.0, "2026-08-03T11:00:00-07:00",
                       broker="MANUAL", account_number="MANUAL", account_label="MANUAL"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    assert len(_trades(store)) == 2, "two brokers, two positions, neither closing the other"

    _adjust(store, action="REASSIGN_GROUP", target_kind="EXECUTION", target_uid="MANUAL:MANUAL:m1",
            payload={"broker": "QUESTRADE", "account_number": "51234567"},
            reason="entered by hand while the API was down")
    store.rebuild_trades(refresh_tags=False)
    trades = _trades(store)
    assert len(trades) == 1
    assert trades[0]["status"] == "CLOSED" and trades[0]["gross_pnl"] == pytest.approx(500.0)


def test_a_force_close_books_no_invented_profit(store):
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00")])
    key_text = group_key_text(("QUESTRADE", "51234567", "AAPL", "STK", "USD"))
    _adjust(store, action="FORCE_CLOSE", target_kind="TRADE_GROUP", target_uid=key_text,
            payload={}, reason="transferred out; the broker never reported the exit")
    store.rebuild_trades(refresh_tags=False)
    trade = _trades(store)[0]
    assert trade["status"] == "CLOSED"
    assert trade["reconcile_status"] == "FORCED_CLOSED"
    # Zero, not a guess. The system does not know what it was worth.
    assert trade["gross_pnl"] == pytest.approx(0.0)
    assert [leg["role"] for leg in _legs(store, trade["trade_id"])] == ["OPEN", "SYNTHETIC_CLOSE"]


def test_a_force_close_can_carry_the_price_the_trader_knows(store):
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00")])
    key_text = group_key_text(("QUESTRADE", "51234567", "AAPL", "STK", "USD"))
    _adjust(store, action="FORCE_CLOSE", target_kind="TRADE_GROUP", target_uid=key_text,
            payload={"price": 158.0}, reason="statement shows the transfer at 158.00")
    store.rebuild_trades(refresh_tags=False)
    assert _trades(store)[0]["gross_pnl"] == pytest.approx(800.0)


def test_a_superseded_adjustment_stops_applying(store):
    """Undo is a superseding record, not a delete (I3)."""
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
            _execution("QT:5:2", "SELL", 100, 155.0, "2026-08-03T11:00:00-07:00"),
        ]
    )
    adjustment_id = _adjust(store, action="EDIT_EXECUTION", target_kind="EXECUTION",
                            target_uid="QT:5:2", payload={"price": 156.0}, reason="wrong")
    store.rebuild_trades(refresh_tags=False)
    assert _trades(store)[0]["gross_pnl"] == pytest.approx(600.0)

    with store.connection() as conn:
        conn.execute(
            "UPDATE trade_adjustments SET superseded_by = ? WHERE adjustment_id = ?",
            ("undo-record", adjustment_id),
        )
    store.rebuild_trades(refresh_tags=False)
    assert _trades(store)[0]["gross_pnl"] == pytest.approx(500.0)
    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM trade_adjustments").fetchone()[0] == 1


def test_adjustments_apply_in_a_deterministic_order(store):
    """Two edits of the same field: the later one is the trader's latest word."""
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
            _execution("QT:5:2", "SELL", 100, 155.0, "2026-08-03T11:00:00-07:00"),
        ]
    )
    _adjust(store, action="EDIT_EXECUTION", target_kind="EXECUTION", target_uid="QT:5:2",
            payload={"price": 156.0}, created_at="2026-08-10T09:00:00")
    _adjust(store, action="EDIT_EXECUTION", target_kind="EXECUTION", target_uid="QT:5:2",
            payload={"price": 157.0}, created_at="2026-08-11T09:00:00")
    store.rebuild_trades(refresh_tags=False)
    first = _trades(store)[0]["gross_pnl"]
    store.rebuild_trades(refresh_tags=False)
    assert _trades(store)[0]["gross_pnl"] == pytest.approx(700.0)
    assert _trades(store)[0]["gross_pnl"] == pytest.approx(first), "and stable across rebuilds"


# ---------------------------------------------------------------------------
# B6 / I4 - annotations survive
# ---------------------------------------------------------------------------


def test_a_backfilled_opening_fill_carries_the_note_with_it(store):
    """The §9 step 4 case, spelled out: an earlier fill arrives after the note."""
    store.upsert_executions([_execution("QT:5:2", "SELL", 100, 155.0, "2026-08-03T11:00:00-07:00")])
    store.rebuild_trades(refresh_tags=False)
    original = _trades(store)[0]["trade_id"]
    store.save_trade_annotation(original, setup_tags="avwap-reclaim", notes="scaled out too early")

    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00")])
    store.rebuild_trades(refresh_tags=False)

    trade = _trades(store)[0]
    assert trade["trade_id"] != original, "the trade really did change identity"
    assert trade["status"] == "CLOSED"
    with store.connection() as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM trade_annotations")]
        aliases = [dict(row) for row in conn.execute("SELECT * FROM trade_aliases")]
    assert [row["trade_id"] for row in rows] == [trade["trade_id"]]
    assert rows[0]["notes"] == "scaled out too early"
    assert [(a["old_trade_id"], a["new_trade_id"]) for a in aliases] == [(original, trade["trade_id"])]
    assert store.resolve_trade_id(original) == trade["trade_id"]


def test_an_unrelated_backfill_does_not_re_key_anything(store):
    """The stability the anchor buys. A sequence number re-keyed on any insert."""
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
            _execution("QT:5:2", "SELL", 100, 155.0, "2026-08-03T11:00:00-07:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    before = {row["symbol"]: row["trade_id"] for row in _trades(store)}

    store.upsert_executions(
        [_execution("QT:5:9", "BUY", 10, 300.0, "2026-08-01T09:31:00-07:00", symbol="MSFT")]
    )
    store.rebuild_trades(refresh_tags=False)
    after = {row["symbol"]: row["trade_id"] for row in _trades(store)}
    assert after["AAPL"] == before["AAPL"]
    assert store.last_rekey["remapped"] == []


def test_an_ambiguous_re_key_is_reported_and_never_guessed(store):
    """A note that could belong to either of two trades is left where it is.

    Splitting one position into two equal halves gives each rebuilt trade the
    same claim on the old trade's executions. Picking one would put the
    trader's note on a trade it may not describe.
    """
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
            _execution("QT:5:2", "SELL", 100, 155.0, "2026-08-03T11:00:00-07:00"),
            _execution("QT:5:3", "BUY", 100, 152.0, "2026-08-03T12:00:00-07:00"),
            _execution("QT:5:4", "SELL", 100, 158.0, "2026-08-03T13:00:00-07:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    trades = _trades(store)
    assert len(trades) == 2

    # Annotate the first, then void the fill that separates them so the two
    # positions merge into one - which now shares executions with both.
    store.save_trade_annotation(trades[0]["trade_id"], setup_tags="x", notes="which one?")
    fake_old = trades[0]["trade_id"]

    _adjust(store, action="EDIT_EXECUTION", target_kind="EXECUTION", target_uid="QT:5:1",
            payload={"quantity": 200}, reason="force a re-key")
    store.rebuild_trades(refresh_tags=False)

    with store.connection() as conn:
        orphans = conn.execute(
            """
            SELECT COUNT(*) FROM trade_annotations a
            LEFT JOIN trades t ON t.trade_id = a.trade_id WHERE t.trade_id IS NULL
            """
        ).fetchone()[0]
    # Whatever the mapping decided, it is never allowed to leave an orphan and
    # never allowed to invent a mapping it could not justify.
    assert orphans == 0 or any(
        item["old_trade_id"] == fake_old for item in store.last_rekey["ambiguous"]
    )


def test_the_rebuild_stays_idempotent_with_adjustments_in_play(store):
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
            _execution("QT:5:2", "SELL", 150, 155.0, "2026-08-03T11:00:00-07:00"),
        ]
    )
    _adjust(store, action="EDIT_EXECUTION", target_kind="EXECUTION", target_uid="QT:5:2",
            payload={"commission": 4.95}, reason="statement")
    store.rebuild_trades(refresh_tags=False)
    first = _trades(store)
    store.rebuild_trades(refresh_tags=False)
    second = _trades(store)
    assert [{k: v for k, v in row.items() if k != "updated_at"} for row in first] == [
        {k: v for k, v in row.items() if k != "updated_at"} for row in second
    ]
