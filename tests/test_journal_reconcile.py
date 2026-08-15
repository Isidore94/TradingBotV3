"""R7 §9 step 9 - does the journal agree with the broker about what is open?

Root cause B1's second half: "no reconciliation against broker positions exists
anywhere". This is also where step 4's deliberate narrowing gets its evidence -
a naked sell is ambiguous to assembly, and unambiguous the moment the broker
says it holds nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_reconcile as jr  # noqa: E402
from journal_identity import group_key_text  # noqa: E402
from journal_store import JournalStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return JournalStore(tmp_path / "trade_journal.sqlite3")


def _execution(uid: str, side: str, quantity: float, price: float, day: str, **overrides):
    row = {
        "execution_uid": uid, "broker": "QUESTRADE", "account_number": "51830546",
        "account_label": "TFSA", "account_type": "", "symbol": "AAPL", "security_type": "STK",
        "currency": "USD", "side": side, "quantity": quantity, "price": price,
        "timestamp": f"{day}T09:31:00-07:00", "trade_date": day, "commission": 0.0, "fees": 0.0,
        "gross_amount": None, "net_amount": None, "order_id": "", "exchange_exec_id": "",
        "raw_json": "{}",
    }
    row.update(overrides)
    return row


def _broker(symbol: str, quantity: float, **overrides):
    row = {
        "broker": "QUESTRADE", "account_number": "51830546", "symbol": symbol,
        "security_type": "STK", "currency": "USD", "quantity": quantity,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


def test_a_position_both_sides_agree_on_is_not_a_mismatch(store):
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03")])
    store.rebuild_trades(refresh_tags=False)
    report = jr.compare(store, [_broker("AAPL", 100)])
    assert report["mismatched"] == []
    assert len(report["agreed"]) == 1


def test_two_open_trades_in_one_instrument_are_one_position(store):
    """Netted, not compared per trade.

    Assembly can legitimately split one holding into two trades; the broker
    reports a single number. Comparing per trade would invent a mismatch out of
    an assembly detail.
    """
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 60, 150.0, "2026-08-03"),
            _execution("QT:5:2", "SELL", 60, 155.0, "2026-08-04"),
            _execution("QT:5:3", "BUY", 100, 152.0, "2026-08-05"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    report = jr.compare(store, [_broker("AAPL", 100)])
    assert report["mismatched"] == []


def test_a_partly_exited_position_reconciles_on_what_is_left(store):
    store.upsert_executions(
        [
            _execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03"),
            _execution("QT:5:2", "SELL", 40, 155.0, "2026-08-04"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        assert conn.execute("SELECT status FROM trades").fetchone()[0] == "CLOSED_PARTIAL"
    assert jr.compare(store, [_broker("AAPL", 60)])["mismatched"] == []


def test_a_short_position_is_compared_with_its_sign(store):
    store.upsert_executions([_execution("QT:5:1", "SELL", 50, 250.0, "2026-08-04", symbol="TSLA")])
    store.rebuild_trades(refresh_tags=False)
    assert jr.compare(store, [_broker("TSLA", -50)])["mismatched"] == []


# ---------------------------------------------------------------------------
# The case step 4 could not judge
# ---------------------------------------------------------------------------


def test_the_naked_sell_step_4_left_alone_is_caught_here(store):
    """The whole reason the step 4 narrowing is safe.

    A sell with no matching buy assembles as an ordinary short, because from the
    execution alone that is indistinguishable from a real short entry. The
    broker reporting flat is the evidence assembly could not have.
    """
    store.upsert_executions([_execution("QT:5:9", "SELL", 100, 95.0, "2026-08-05", symbol="AMD")])
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        row = dict(conn.execute("SELECT direction, reconcile_status FROM trades").fetchone())
    assert row == {"direction": "SHORT", "reconcile_status": ""}, "step 4 said nothing, by design"

    report = jr.reconcile(store, [], brokers=["QUESTRADE"])
    assert [item["kind"] for item in report["mismatched"]] == ["JOURNAL_OPEN_BROKER_FLAT"]
    with store.connection() as conn:
        assert conn.execute("SELECT reconcile_status FROM trades").fetchone()[0] == "NEEDS_REVIEW"


def test_a_journal_open_broker_flat_position_only_ever_suggests(store):
    """It never closes anything. An adjustment is a thing a human decided."""
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03")])
    store.rebuild_trades(refresh_tags=False)
    report = jr.reconcile(store, [], brokers=["QUESTRADE"])

    assert len(report["suggestions"]) == 1
    suggestion = report["suggestions"][0]
    assert suggestion["action"] == "FORCE_CLOSE"
    assert suggestion["target_uid"] == group_key_text(("QUESTRADE", "51830546", "AAPL", "STK", "USD"))

    # Nothing applied: no adjustment exists, and the trade is still open.
    assert store.list_adjustments() == []
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        assert conn.execute("SELECT status FROM trades").fetchone()[0] == "OPEN"


def test_confirming_a_suggestion_is_what_closes_it(store):
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03")])
    store.rebuild_trades(refresh_tags=False)
    report = jr.reconcile(store, [], brokers=["QUESTRADE"])

    jr.confirm_suggestion(store, report["suggestions"][0], reason="transferred out, confirmed")
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        row = dict(conn.execute("SELECT status, reconcile_status FROM trades").fetchone())
    assert row == {"status": "CLOSED", "reconcile_status": "FORCED_CLOSED"}
    assert store.list_adjustments()[0]["reason"] == "transferred out, confirmed"


# ---------------------------------------------------------------------------
# The other two mismatch shapes
# ---------------------------------------------------------------------------


def test_a_position_the_broker_holds_and_the_journal_does_not_is_a_mismatch(store):
    """The literal "missing trades" case, finally visible."""
    report = jr.compare(store, [_broker("NVDA", 25)])
    assert [item["kind"] for item in report["mismatched"]] == ["BROKER_OPEN_JOURNAL_FLAT"]
    assert report["mismatched"][0]["delta"] == pytest.approx(25.0)
    assert report["suggestions"] == [], "nothing to force-close; the fix is an import"


def test_a_quantity_disagreement_is_its_own_kind(store):
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03")])
    store.rebuild_trades(refresh_tags=False)
    report = jr.compare(store, [_broker("AAPL", 60)])
    assert [item["kind"] for item in report["mismatched"]] == ["QUANTITY_MISMATCH"]
    assert report["mismatched"][0]["delta"] == pytest.approx(-40.0)


def test_dust_is_not_a_mismatch(store):
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03")])
    store.rebuild_trades(refresh_tags=False)
    assert jr.compare(store, [_broker("AAPL", 100.00001)])["mismatched"] == []


def test_scoping_to_one_broker_does_not_accuse_the_other(store):
    """An IBKR-only check must not report every Questrade position as flat."""
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03")])
    store.rebuild_trades(refresh_tags=False)
    assert jr.compare(store, [], brokers=["IBKR"])["mismatched"] == []
    assert jr.compare(store, [], brokers=["QUESTRADE"])["mismatched"] != []


# ---------------------------------------------------------------------------
# Flags, records, reports
# ---------------------------------------------------------------------------


def test_a_position_that_reconciles_today_loses_yesterdays_flag(store):
    """Or the review queue only grows and stops meaning anything."""
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03")])
    store.rebuild_trades(refresh_tags=False)
    jr.reconcile(store, [], brokers=["QUESTRADE"])
    with store.connection() as conn:
        assert conn.execute("SELECT reconcile_status FROM trades").fetchone()[0] == "NEEDS_REVIEW"

    jr.reconcile(store, [_broker("AAPL", 100)], brokers=["QUESTRADE"])
    with store.connection() as conn:
        assert conn.execute("SELECT reconcile_status FROM trades").fetchone()[0] == ""


def test_reconciliation_never_overwrites_a_human_decision(store):
    """FORCED_CLOSED records that a human closed it. This module is not a human."""
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03")])
    key_text = group_key_text(("QUESTRADE", "51830546", "AAPL", "STK", "USD"))
    store.record_adjustment(action="FORCE_CLOSE", target_uid=key_text, reason="transferred out")
    store.rebuild_trades(refresh_tags=False)

    jr.reconcile(store, [_broker("AAPL", 100)], brokers=["QUESTRADE"])
    with store.connection() as conn:
        assert conn.execute("SELECT reconcile_status FROM trades").fetchone()[0] == "FORCED_CLOSED"


def test_every_run_leaves_an_append_only_record(store):
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03")])
    store.rebuild_trades(refresh_tags=False)
    jr.reconcile(store, [], brokers=["QUESTRADE"], trigger="nightly")
    runs = [row for row in store.list_import_runs() if row["source"] == "RECONCILE"]
    assert len(runs) == 1
    assert runs[0]["status"] == "MISMATCH" and runs[0]["trigger"] == "nightly"
    assert "1 mismatch(es)" in runs[0]["message"]


def test_the_report_is_readable_afterwards_without_re_running(store):
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03")])
    store.rebuild_trades(refresh_tags=False)
    written = jr.reconcile(store, [], brokers=["QUESTRADE"])
    read_back = jr.last_report(store)
    assert read_back["positions_checked"] == written["positions_checked"]
    assert len(read_back["suggestions"]) == 1


def test_a_suggestion_is_stored_where_a_rebuild_cannot_reach_it(store):
    """Stored in meta, not in trade_adjustments - or it would apply itself."""
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03")])
    store.rebuild_trades(refresh_tags=False)
    jr.reconcile(store, [], brokers=["QUESTRADE"])
    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM trade_adjustments").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM meta WHERE key = ?", (jr.REPORT_META_KEY,)
        ).fetchone()[0] == 1


def test_a_clean_reconciliation_reports_ok(store):
    store.upsert_executions([_execution("QT:5:1", "BUY", 100, 150.0, "2026-08-03")])
    store.rebuild_trades(refresh_tags=False)
    report = jr.reconcile(store, [_broker("AAPL", 100)], brokers=["QUESTRADE"])
    assert report["mismatched"] == [] and report["flagged_trades"] == 0
    runs = [row for row in store.list_import_runs() if row["source"] == "RECONCILE"]
    assert runs[0]["status"] == "OK"
