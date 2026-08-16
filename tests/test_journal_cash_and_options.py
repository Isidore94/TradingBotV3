"""R7 §9 step 7 - the money and the option events that never reached the journal.

Root cause A7. ``get_activities`` was written and never called, and the Flex
parser read only Trade/TradeConfirm - so dividends, interest, fees, FX and every
option expiry, exercise and assignment were invisible.

The expiry gap is the one the trader sees as a stuck-open trade: an option that
expired worthless has no trade, therefore no closing fill, therefore a position
nothing can ever close.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_coverage as jc  # noqa: E402
import journal_runner  # noqa: E402
from journal_importers import (  # noqa: E402
    classify_activity_type,
    flex_cash_transactions,
    flex_open_positions,
    flex_option_eae_executions,
    normalize_questrade_activity,
    questrade_trade_activity_dates,
)
from journal_store import JournalStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return JournalStore(tmp_path / "trade_journal.sqlite3")


# ---------------------------------------------------------------------------
# Option lifecycle: the stuck-open option finally closes
# ---------------------------------------------------------------------------


def _eae(quantity: float, **overrides):
    row = {
        "accountId": "U4867396",
        "symbol": "SPY260116C00500000",
        "assetCategory": "OPT",
        "currency": "USD",
        "date": "20260116;163000",
        "transactionType": "Expiration",
        "quantity": quantity,
        "tradePrice": "0",
        "transactionID": "eae-1",
        "multiplier": "100",
    }
    row.update(overrides)
    return row


def test_an_expired_option_becomes_the_closing_fill_it_never_had(store):
    """The concrete stuck-open case, closed end to end."""
    store.upsert_executions(
        [
            {
                "execution_uid": "IBKR:U4867396:x1", "broker": "IBKR", "account_number": "U4867396",
                "account_label": "U4867396", "account_type": "", "symbol": "SPY260116C00500000",
                "security_type": "OPT", "currency": "USD", "side": "BUY", "quantity": 2,
                "price": 5.0, "timestamp": "2026-08-04T06:40:00-07:00", "trade_date": "2026-08-04",
                "commission": 1.3, "fees": 0.0, "gross_amount": None, "net_amount": None,
                "order_id": "", "exchange_exec_id": "", "raw_json": json.dumps({"multiplier": "100"}),
            }
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        assert conn.execute("SELECT status FROM trades").fetchone()[0] == "OPEN"

    store.upsert_executions(flex_option_eae_executions([_eae(-2)]))
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        row = dict(conn.execute("SELECT status, gross_pnl, quantity_closed FROM trades").fetchone())
    assert row["status"] == "CLOSED"
    assert row["quantity_closed"] == 2.0
    # Expired worthless: the whole premium is the loss, at the x100 multiplier.
    assert row["gross_pnl"] == pytest.approx(-1000.0)


def test_the_side_follows_the_sign_so_a_short_option_closes_too(store):
    """A negative quantity reduces the position from whichever side it was held."""
    closing = flex_option_eae_executions([_eae(-2)])[0]
    opening = flex_option_eae_executions([_eae(2)])[0]
    assert closing.side == "SELL" and opening.side == "BUY"
    assert closing.quantity == 2.0


def test_an_assignment_is_recorded_at_its_price_not_at_zero():
    execution = flex_option_eae_executions(
        [_eae(-1, transactionType="Assignment", tradePrice="3.25")]
    )[0]
    assert execution.price == pytest.approx(3.25)
    assert execution.security_type == "OPT"


def test_the_same_option_event_imported_twice_is_one_execution(store):
    store.upsert_executions(flex_option_eae_executions([_eae(-2)]))
    store.upsert_executions(flex_option_eae_executions([_eae(-2)]))
    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_executions").fetchone()[0] == 1


def test_an_unreadable_option_event_is_quarantined_not_stamped_now():
    quarantine: list = []
    executions = flex_option_eae_executions([_eae(-2, date="sometime")], quarantine=quarantine)
    assert executions == []
    assert len(quarantine) == 1


def test_a_zero_quantity_event_is_not_a_fill():
    assert flex_option_eae_executions([_eae(0)]) == []


# ---------------------------------------------------------------------------
# Cash transactions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Dividends", "DIVIDEND"), ("Interest", "INTEREST"), ("FX conversion", "FX"),
        ("Fees and rebates", "FEE"), ("Broker Interest Paid", "INTEREST"),
        ("Withholding Tax", "FEE"), ("Payment In Lieu Of Dividends", "DIVIDEND"),
        ("Some New Thing", "OTHER"), ("", "OTHER"),
    ],
)
def test_both_brokers_activity_words_land_in_one_vocabulary(label, expected):
    assert classify_activity_type(label) == expected


def test_a_questrade_dividend_becomes_a_cash_row(store):
    row = normalize_questrade_activity(
        {
            "type": "Dividends", "symbol": "MSFT", "netAmount": 12.5, "currency": "USD",
            "settlementDate": "2026-08-05T00:00:00-04:00", "description": "MSFT dividend",
        },
        {"number": "51830546", "type": "TFSA"},
    )
    store.upsert_cash_transactions([row])
    stored = store.list_cash_transactions()
    assert len(stored) == 1
    assert stored[0]["activity_type"] == "DIVIDEND"
    assert stored[0]["amount"] == pytest.approx(12.5)
    assert stored[0]["txn_date"] == "2026-08-05"


def test_the_same_dividend_imported_every_night_stays_one_row(store):
    """Neither broker supplies a stable id, so the uid has to be derived."""
    raw = {
        "type": "Dividends", "symbol": "MSFT", "netAmount": 12.5, "currency": "USD",
        "settlementDate": "2026-08-05", "description": "MSFT dividend",
    }
    account = {"number": "51830546"}
    for _ in range(3):
        store.upsert_cash_transactions([normalize_questrade_activity(raw, account)])
    assert len(store.list_cash_transactions()) == 1


def test_a_trade_activity_is_never_imported_as_cash():
    """The executions endpoint is authoritative; importing here would double-count."""
    assert normalize_questrade_activity({"type": "Trades", "symbol": "AAPL"}, {"number": "5"}) is None


def test_a_flex_cash_transaction_becomes_a_cash_row(store):
    rows = flex_cash_transactions(
        [
            {
                "accountId": "U4867396", "type": "Dividends", "amount": "12.50",
                "currency": "USD", "dateTime": "20260805", "symbol": "MSFT",
                "description": "MSFT dividend", "transactionID": "t1",
            }
        ]
    )
    store.upsert_cash_transactions(rows)
    stored = store.list_cash_transactions(broker="IBKR")
    assert stored[0]["activity_type"] == "DIVIDEND" and stored[0]["amount"] == pytest.approx(12.5)


def test_a_flex_cash_transaction_accepts_the_real_compact_timestamp(store):
    rows = flex_cash_transactions(
        [
            {
                "accountId": "U4867396", "type": "Dividends", "amount": "12.50",
                "currency": "USD", "dateTime": "20260805;093211", "symbol": "MSFT",
                "description": "MSFT dividend", "transactionID": "t-compact",
            }
        ]
    )

    assert len(rows) == 1
    assert rows[0]["txn_date"] == "2026-08-05"
    store.upsert_cash_transactions(rows)
    assert store.list_cash_transactions(broker="IBKR")[0]["txn_uid"] == rows[0]["txn_uid"]


def test_cash_rows_never_reach_assembly(store):
    """A dividend moves money and is not a trade. One in raw_executions would
    invent a position out of it."""
    store.upsert_cash_transactions(
        flex_cash_transactions(
            [{"accountId": "U1", "type": "Dividends", "amount": "12.50", "currency": "USD",
              "dateTime": "20260805", "symbol": "MSFT", "transactionID": "t1"}]
        )
    )
    assert store.rebuild_trades(refresh_tags=False) == 0
    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_executions").fetchone()[0] == 0


def test_cash_rows_can_be_filtered_for_the_fees_view(store):
    store.upsert_cash_transactions(
        flex_cash_transactions(
            [
                {"accountId": "U1", "type": "Other Fees", "amount": "-3.00", "currency": "USD",
                 "dateTime": "20260805", "transactionID": "f1"},
                {"accountId": "U1", "type": "Dividends", "amount": "12.50", "currency": "USD",
                 "dateTime": "20260806", "transactionID": "d1"},
            ]
        )
    )
    assert len(store.list_cash_transactions(activity_type="FEE")) == 1
    assert len(store.list_cash_transactions(date_from="2026-08-06")) == 1


# ---------------------------------------------------------------------------
# Open positions: input to reconciliation, never executions
# ---------------------------------------------------------------------------


def test_open_positions_are_read_but_never_stored_as_executions(store):
    positions = flex_open_positions(
        [{"accountId": "U4867396", "symbol": "MSFT", "position": "100", "assetCategory": "STK",
          "currency": "USD"}]
    )
    assert positions[0]["quantity"] == pytest.approx(100.0)
    assert positions[0]["security_type"] == "STK"
    # Nothing in this module writes them anywhere. A position is a statement of
    # where things stand, not a record of a trade; as a synthetic fill it would
    # corrupt the P&L it exists to check.
    assert store.rebuild_trades(refresh_tags=False) == 0


# ---------------------------------------------------------------------------
# The completeness cross-check
# ---------------------------------------------------------------------------


def test_trade_activity_days_are_extracted_for_the_cross_check():
    days = questrade_trade_activity_dates(
        [
            {"type": "Trades", "tradeDate": "2026-08-05T09:31:00-07:00"},
            {"type": "Dividends", "tradeDate": "2026-08-06T00:00:00-07:00"},
        ]
    )
    assert days == {date(2026, 8, 5)}


class _ActivityImporter:
    """Executions return nothing; activities insist a trade happened."""

    refresh_token = "token"
    access_token = ""
    api_server = ""

    def __init__(self):
        self.quarantined = []

    def iter_execution_chunks(self, start_date, end_date):
        yield {
            "account": {"number": "51830546", "type": "TFSA"},
            "account_number": "51830546",
            "start": start_date,
            "end": end_date,
            "executions": [],
        }

    def get_activities(self, account_number, start, end):
        return [
            {"type": "Trades", "tradeDate": "2026-08-05T09:31:00-07:00", "symbol": "AAPL"},
            {"type": "Dividends", "settlementDate": "2026-08-05", "symbol": "MSFT",
             "netAmount": 1.0, "currency": "USD", "description": "d"},
        ]


def test_a_day_the_activities_endpoint_calls_traded_is_not_left_covered(store, monkeypatch):
    """The cross-check earns its keep: executions said nothing, activities disagree.

    Marking the day COVERED would record "we looked, there was nothing" for a day
    the broker's own activity feed says had a trade. That is exactly the silent
    hole I2 exists to prevent.
    """
    monkeypatch.setattr(journal_runner, "QuestradeImporter", _ActivityImporter)
    result = journal_runner.run_journal_backfill(
        days=10, store=store, include_ibkr_flex=False
    )
    rows = {row["day"]: row["status"] for row in jc.coverage_rows(store, broker="QUESTRADE")}
    assert rows["2026-08-05"] == jc.FAILED
    assert any("executions did not return" in message for message in result["messages"])
    # And the dividend still landed, because activities are additive.
    assert len(store.list_cash_transactions()) == 1


def test_a_broken_activities_endpoint_preserves_rows_but_not_green_coverage(store, monkeypatch):
    class _Broken(_ActivityImporter):
        def get_activities(self, account_number, start, end):
            raise RuntimeError("activities 500")

    monkeypatch.setattr(journal_runner, "QuestradeImporter", _Broken)
    result = journal_runner.run_journal_backfill(days=3, store=store, include_ibkr_flex=False)
    assert result["status"] == "FAILED"
    assert any("activities" in message and "skipped" in message for message in result["messages"])
    statuses = {row["status"] for row in jc.coverage_rows(store, broker="QUESTRADE")}
    assert jc.FAILED in statuses and jc.COVERED not in statuses, (
        "without the independent cross-check completeness is unknown"
    )
