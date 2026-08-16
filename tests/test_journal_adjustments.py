"""R7 §9 step 5 - the write side of trader corrections (root cause B7, I3).

Step 4 taught assembly to apply ``trade_adjustments``. This is how one gets
there, and the rules it enforces are the ones that make the table an audit
trail rather than a scratchpad: append-only, a mandatory reason, an action the
system will actually honour, and undo by superseding rather than deleting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from journal_identity import group_key_text  # noqa: E402
from journal_store import JournalStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return JournalStore(tmp_path / "trade_journal.sqlite3")


def _execution(uid: str, side: str, quantity: float, price: float, timestamp: str, **overrides):
    row = {
        "execution_uid": uid, "broker": "QUESTRADE", "account_number": "51234567",
        "account_label": "Margin", "account_type": "", "symbol": "AAPL", "security_type": "STK",
        "currency": "USD", "side": side, "quantity": quantity, "price": price,
        "timestamp": timestamp, "trade_date": timestamp[:10], "commission": 0.0, "fees": 0.0,
        "gross_amount": None, "net_amount": None, "order_id": "", "exchange_exec_id": "",
        "raw_json": "{}",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# What the store refuses to accept
# ---------------------------------------------------------------------------


def test_a_correction_without_a_reason_is_refused(store):
    """Six months later, an unexplained correction is indistinguishable from a mistake."""
    with pytest.raises(ValueError, match="reason is required"):
        store.record_adjustment(action="VOID_EXECUTION", target_uid="QT:5:1", reason="")
    with pytest.raises(ValueError, match="reason is required"):
        store.record_adjustment(action="VOID_EXECUTION", target_uid="QT:5:1", reason="   ")


def test_an_unknown_action_is_refused_at_write_time(store):
    """Not silently ignored at rebuild time.

    A correction the trader believes they made, which quietly does nothing, is
    worse than one that was never accepted - they would find out from a wrong
    tax number, not from an error.
    """
    with pytest.raises(ValueError, match="unsupported adjustment action"):
        store.record_adjustment(action="DELETE_TRADE", target_uid="QT:5:1", reason="nope")


def test_an_action_cannot_be_pointed_at_the_wrong_kind_of_target(store):
    with pytest.raises(ValueError, match="targets EXECUTION"):
        store.record_adjustment(
            action="VOID_EXECUTION", target_kind="TRADE_GROUP", target_uid="x", reason="wrong kind"
        )
    with pytest.raises(ValueError, match="targets TRADE_GROUP"):
        store.record_adjustment(
            action="FORCE_CLOSE", target_kind="EXECUTION", target_uid="x", reason="wrong kind"
        )


def test_a_blank_target_is_refused(store):
    with pytest.raises(ValueError, match="target_uid is required"):
        store.record_adjustment(action="VOID_EXECUTION", target_uid="  ", reason="something")


def test_superseding_an_unknown_record_is_refused_and_writes_nothing(store):
    with pytest.raises(ValueError, match="cannot supersede unknown adjustment"):
        store.record_adjustment(
            action="VOID_EXECUTION", target_uid="QT:5:1", reason="undo", supersedes="not-a-real-id"
        )
    assert store.list_adjustments() == [], "a refused write leaves no half-record behind"


# ---------------------------------------------------------------------------
# What it records
# ---------------------------------------------------------------------------


def test_each_action_defaults_to_its_own_target_kind(store):
    for action, expected in [
        ("VOID_EXECUTION", "EXECUTION"),
        ("EDIT_EXECUTION", "EXECUTION"),
        ("ADD_EXECUTION", "EXECUTION"),
        ("REASSIGN_GROUP", "EXECUTION"),
        ("FORCE_CLOSE", "TRADE_GROUP"),
    ]:
        record = store.record_adjustment(action=action, target_uid="target", reason="because")
        assert record["target_kind"] == expected


def test_the_record_keeps_its_payload_and_its_reason(store):
    record = store.record_adjustment(
        action="EDIT_EXECUTION",
        target_uid="QT:5:2",
        payload={"price": 156.0},
        reason="the statement says 156.00",
        source="journal-health",
    )
    stored = store.list_adjustments()[0]
    assert stored["adjustment_id"] == record["adjustment_id"]
    assert stored["payload"] == {"price": 156.0}
    assert stored["reason"] == "the statement says 156.00"
    assert stored["source"] == "journal-health"
    assert stored["superseded_by"] == ""


def test_two_corrections_never_share_an_id(store):
    ids = {
        store.record_adjustment(action="VOID_EXECUTION", target_uid="QT:5:1", reason=f"n{i}")[
            "adjustment_id"
        ]
        for i in range(5)
    }
    assert len(ids) == 5


# ---------------------------------------------------------------------------
# Undo, the append-only way (I3)
# ---------------------------------------------------------------------------


def test_undo_supersedes_and_never_deletes(store):
    original = store.record_adjustment(
        action="VOID_EXECUTION", target_uid="QT:5:dupe", reason="broker sent it twice"
    )
    undo = store.undo_adjustment(original["adjustment_id"], reason="it was not a duplicate after all")

    everything = store.list_adjustments()
    assert len(everything) == 2, "the history keeps both the correction and its undo"
    by_id = {row["adjustment_id"]: row for row in everything}
    assert by_id[original["adjustment_id"]]["superseded_by"] == undo["adjustment_id"]
    assert by_id[original["adjustment_id"]]["reason"] == "broker sent it twice"

    active = store.list_active_adjustments()
    assert [row["adjustment_id"] for row in active] == [undo["adjustment_id"]]


def test_an_undo_changes_nothing_by_itself(store):
    """The superseding record carries an empty payload on purpose."""
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
            _execution("QT:5:2", "SELL", 100, 155.0, "2026-08-03T11:00:00-07:00"),
        ]
    )
    edit = store.record_adjustment(
        action="EDIT_EXECUTION", target_uid="QT:5:2", payload={"price": 160.0}, reason="wrong"
    )
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        assert conn.execute("SELECT gross_pnl FROM trades").fetchone()[0] == pytest.approx(1000.0)

    store.undo_adjustment(edit["adjustment_id"], reason="the statement agreed with the broker")
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        assert conn.execute("SELECT gross_pnl FROM trades").fetchone()[0] == pytest.approx(500.0)


def test_undoing_an_undo_is_another_record(store):
    first = store.record_adjustment(action="VOID_EXECUTION", target_uid="QT:5:1", reason="one")
    second = store.undo_adjustment(first["adjustment_id"], reason="two")
    third = store.undo_adjustment(second["adjustment_id"], reason="three")
    assert len({first["adjustment_id"], second["adjustment_id"], third["adjustment_id"]}) == 3
    assert len(store.list_adjustments()) == 3
    assert [row["adjustment_id"] for row in store.list_active_adjustments()] == [third["adjustment_id"]]


def test_a_force_close_can_be_undone_and_the_position_reopens(store):
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00")])
    key_text = group_key_text(("QUESTRADE", "51234567", "AAPL", "STK", "USD"))
    forced = store.record_adjustment(
        action="FORCE_CLOSE", target_uid=key_text, payload={"price": 158.0},
        reason="transferred out",
    )
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        row = dict(conn.execute("SELECT status, reconcile_status FROM trades").fetchone())
    assert (row["status"], row["reconcile_status"]) == ("CLOSED", "FORCED_CLOSED")

    undone = store.undo_adjustment(forced["adjustment_id"], reason="the shares came back")
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        row = dict(conn.execute("SELECT status, reconcile_status FROM trades").fetchone())
    assert (row["status"], row["reconcile_status"]) == ("OPEN", "")

    reapplied = store.undo_adjustment(undone["adjustment_id"], reason="transfer was confirmed")
    assert reapplied["action"] == "FORCE_CLOSE"
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        row = dict(conn.execute("SELECT status, reconcile_status FROM trades").fetchone())
    assert (row["status"], row["reconcile_status"]) == ("CLOSED", "FORCED_CLOSED")


# ---------------------------------------------------------------------------
# Reading the trail
# ---------------------------------------------------------------------------


def test_the_trail_can_be_read_per_target_and_filtered_to_what_still_applies(store):
    store.record_adjustment(action="VOID_EXECUTION", target_uid="QT:5:1", reason="a")
    kept = store.record_adjustment(action="VOID_EXECUTION", target_uid="QT:5:2", reason="b")
    store.undo_adjustment(
        store.list_adjustments(target_uid="QT:5:1")[0]["adjustment_id"], reason="undo a"
    )

    assert {row["target_uid"] for row in store.list_adjustments(target_uid="QT:5:1")} == {"QT:5:1"}
    active_targets = {row["target_uid"] for row in store.list_adjustments(include_superseded=False)}
    assert active_targets == {"QT:5:1", "QT:5:2"}, "the undo record itself still targets QT:5:1"
    assert kept["adjustment_id"] in {row["adjustment_id"] for row in store.list_active_adjustments()}


def test_the_trail_is_newest_first(store):
    store.record_adjustment(action="VOID_EXECUTION", target_uid="QT:5:1", reason="first")
    latest = store.record_adjustment(action="VOID_EXECUTION", target_uid="QT:5:2", reason="second")
    assert store.list_adjustments()[0]["adjustment_id"] == latest["adjustment_id"]


def test_a_corrupt_payload_reads_as_empty_rather_than_raising(store):
    """One unreadable row must not take the whole Health tab down with it."""
    record = store.record_adjustment(action="VOID_EXECUTION", target_uid="QT:5:1", reason="x")
    with store.connection() as conn:
        conn.execute(
            "UPDATE trade_adjustments SET payload_json = ? WHERE adjustment_id = ?",
            ("{not json", record["adjustment_id"]),
        )
    assert store.list_adjustments()[0]["payload"] == {}
    assert store.list_active_adjustments()[0]["payload"] == {}


def test_a_recorded_correction_survives_the_next_import(store):
    """The whole point of B7: a correction is not a one-time edit.

    An import re-upserts the row the trader corrected. Because the correction
    lives as a record and is re-applied at every rebuild, the import cannot
    quietly undo it.
    """
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03T09:31:00-07:00"),
            _execution("QT:5:2", "SELL", 100, 155.0, "2026-08-03T11:00:00-07:00"),
        ]
    )
    store.record_adjustment(
        action="EDIT_EXECUTION", target_uid="QT:5:2", payload={"price": 156.0},
        reason="the statement says 156.00",
    )
    store.rebuild_trades(refresh_tags=False)

    # The nightly import re-fetches the same day and writes the broker's value
    # back over the row.
    store.upsert_executions([_execution("QT:5:2", "SELL", 100, 155.0, "2026-08-03T11:00:00-07:00")])
    store.rebuild_trades(refresh_tags=False)

    with store.connection() as conn:
        assert conn.execute("SELECT gross_pnl FROM trades").fetchone()[0] == pytest.approx(600.0)
