"""R7 §9 step 3 - what makes two fills the same instrument (root cause B3).

The defect was never in one place: Questrade fell back to ``listingExchange``
when it had no ``securityType``, the socket and Flex spell IBKR's own categories
differently, and the two brokers use different words for the same thing. All
three reached the group key un-normalized, and any disagreement split one
position into two that could never net.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from journal_identity import (  # noqa: E402
    CANONICAL_SECURITY_TYPES,
    SECURITY_TYPE_ALIASES,
    group_key,
    normalize_security_type,
)
from journal_importers import IBKRExecutionImporter, QuestradeImporter, parse_ibkr_flex_statement  # noqa: E402
from journal_store import JournalStore  # noqa: E402


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        # The two brokers' words for the same instrument.
        ("Stock", "STK"), ("STK", "STK"), ("STOCK", "STK"), ("Equity", "STK"), ("ETF", "STK"),
        ("Option", "OPT"), ("OPT", "OPT"), ("EquityOption", "OPT"),
        ("Future", "FUT"), ("FUT", "FUT"), ("FuturesOption", "FOP"),
        ("Cash", "CASH"), ("Forex", "CASH"),
        ("Bond", "BOND"), ("MutualFund", "FUND"),
        # Spacing and separators brokers sprinkle in.
        ("equity option", "OPT"), ("mutual_fund", "FUND"), ("futures-option", "FOP"),
    ],
)
def test_both_brokers_spellings_land_on_one_word(spelling, expected):
    assert normalize_security_type(spelling) == expected


@pytest.mark.parametrize("exchange", ["NASDAQ", "NYSE", "ARCA", "TSX", "OTC", "nasdaq"])
def test_a_listing_exchange_resolves_to_stock(exchange):
    """The exact defect: an exchange name reached the key as a security type.

    Questrade sends a real ``securityType`` for options and futures, so the
    fallback only ever fired for plain stock - which is why this mapping is
    safe, and why it is an explicit list rather than a heuristic.
    """
    assert normalize_security_type(exchange) == "STK"


def test_an_unrecognized_type_is_kept_apart_rather_than_merged():
    """Between a silent wrong number and a visible wrong shape, take the visible one.

    Folding the unknown into UNKNOWN would merge two genuinely different
    instruments that share a symbol and produce a wrong P&L with no sign of it.
    Leaving them apart produces a stuck-open pair the trader can see and fix
    with a REASSIGN_GROUP adjustment.
    """
    assert normalize_security_type("SomeNewProduct2027") == "SOMENEWPRODUCT2027"
    assert normalize_security_type("") == "UNKNOWN"
    assert normalize_security_type(None) == "UNKNOWN"


def test_every_alias_lands_inside_the_canonical_vocabulary():
    assert set(SECURITY_TYPE_ALIASES.values()) <= CANONICAL_SECURITY_TYPES


def test_the_group_key_normalizes_the_stored_row_not_just_new_imports():
    """A fix that only applied at import time would leave every stuck position stuck."""
    stored_stock = {"broker": "QUESTRADE", "account_number": "5", "symbol": "AMZN",
                    "security_type": "STOCK", "currency": "USD"}
    stored_exchange = {**stored_stock, "security_type": "NASDAQ"}
    assert group_key(stored_stock) == group_key(stored_exchange)


def test_an_option_is_never_grouped_with_its_underlying():
    stock = {"broker": "IBKR", "account_number": "U1", "symbol": "SPY",
             "security_type": "STK", "currency": "USD"}
    option = {**stock, "symbol": "SPY260116C00500000", "security_type": "OPT"}
    assert group_key(stock) != group_key(option)


def test_questrade_no_longer_files_an_option_under_its_stock():
    """``underlyingSymbol`` named the stock, and was a symbol fallback."""
    importer = QuestradeImporter.__new__(QuestradeImporter)
    importer.quarantined = []
    execution = importer.normalize_execution(
        {
            "id": "e1",
            "underlyingSymbol": "AAPL",
            "securityType": "Option",
            "quantity": 1,
            "price": 5.0,
            "timestamp": "2026-08-03T09:31:00-07:00",
        },
        {"number": "51234567"},
    )
    assert execution.symbol != "AAPL"
    assert execution.security_type == "OPT"


def test_questrade_no_longer_calls_an_exchange_a_security_type():
    importer = QuestradeImporter.__new__(QuestradeImporter)
    importer.quarantined = []
    execution = importer.normalize_execution(
        {
            "id": "e1",
            "symbol": "AMZN",
            "listingExchange": "NASDAQ",
            "quantity": 25,
            "price": 180.0,
            "timestamp": "2026-08-05T09:35:00-07:00",
        },
        {"number": "51234567"},
    )
    assert execution.security_type == "UNKNOWN", (
        "an absent securityType is unknown, and unknown is not 'NASDAQ'"
    )


def test_flex_and_the_socket_agree_on_one_vocabulary():
    xml = """
    <FlexQueryResponse><FlexStatements><FlexStatement><Trades>
      <Trade accountId="U1" symbol="AAPL" dateTime="20260804;093100" quantity="10"
             tradePrice="150" buySell="BUY" ibExecID="x1" assetCategory="STK"/>
      <Trade accountId="U1" symbol="SPY260116C00500000" dateTime="20260804;093100" quantity="1"
             tradePrice="5" buySell="BUY" ibExecID="x2" assetCategory="OPT"/>
    </Trades></FlexStatement></FlexStatements></FlexQueryResponse>
    """
    types = {item.security_type for item in parse_ibkr_flex_statement(xml)}
    assert types == {"STK", "OPT"}
    assert types <= CANONICAL_SECURITY_TYPES


def test_flex_and_socket_option_spellings_land_on_the_same_group():
    socket_importer = IBKRExecutionImporter.__new__(IBKRExecutionImporter)
    socket_importer.commissions = {}
    socket_row = socket_importer.normalize_execution(
        SimpleNamespace(
            localSymbol="SPY   260116C00500000", symbol="SPY", secType="OPT",
            lastTradeDateOrContractMonth="20260116", strike=500.0, right="CALL",
            currency="USD", exchange="SMART",
        ),
        SimpleNamespace(
            time="20260804 09:31:00 US/Pacific", execId="socket-1", acctNumber="U1",
            side="BOT", shares=1, price=5.0, orderId="1", permId="2",
        ),
    )
    flex_row = parse_ibkr_flex_statement(
        """
        <FlexQueryResponse><FlexStatements><FlexStatement><Trades>
          <Trade accountId="U1" symbol="SPY260116C00500000" underlyingSymbol="SPY"
                 expiry="20260116" strike="500" putCall="Call"
                 dateTime="20260804;093100" quantity="1" tradePrice="5"
                 buySell="BUY" ibExecID="flex-1" assetCategory="OPT" currency="USD"/>
        </Trades></FlexStatement></FlexStatements></FlexQueryResponse>
        """
    )[0]

    assert socket_row.symbol == flex_row.symbol == "SPY260116C00500000"
    assert group_key(socket_row.__dict__) == group_key(flex_row.__dict__)


def test_the_split_amzn_position_nets_after_a_rebuild(tmp_path):
    """End to end, on stored v2-shaped rows: two stuck-open halves become one trade."""
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    common = dict(
        broker="QUESTRADE", account_number="51234567", account_label="Margin", account_type="",
        symbol="AMZN", currency="USD", trade_date="2026-08-05", commission=4.95, fees=0.0,
        gross_amount=None, net_amount=None, order_id="", exchange_exec_id="", raw_json="{}",
    )
    store.upsert_executions(
        [
            dict(common, execution_uid="QT:51234567:a1", security_type="STOCK", side="BUY",
                 quantity=25, price=180.0, timestamp="2026-08-05T09:35:00-07:00"),
            dict(common, execution_uid="QT:51234567:a2", security_type="NASDAQ", side="SELL",
                 quantity=25, price=184.0, timestamp="2026-08-05T11:05:00-07:00"),
        ]
    )
    assert store.rebuild_trades(refresh_tags=False) == 1
    with store.connection() as conn:
        row = dict(conn.execute("SELECT * FROM trades").fetchone())
    assert (row["status"], row["direction"], row["security_type"]) == ("CLOSED", "LONG", "STK")
    assert row["gross_pnl"] == pytest.approx(100.0)


def test_a_merged_group_receives_its_fills_in_time_order(tmp_path):
    """The subtle break a merge could have introduced, pinned.

    Rows used to be loaded ordered by the *un-normalized* security_type, so a
    group that only exists because two spellings merged would have been handed
    its fills in spelling order. Assembly nets in the order it is given, so the
    sell below would have arrived before the buy that funds it.
    """
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    common = dict(
        broker="QUESTRADE", account_number="5", account_label="M", account_type="",
        symbol="AMZN", currency="USD", trade_date="2026-08-05", commission=0.0, fees=0.0,
        gross_amount=None, net_amount=None, order_id="", exchange_exec_id="", raw_json="{}",
    )
    store.upsert_executions(
        [
            # "STOCK" sorts before "NASDAQ"? No - "NASDAQ" < "STOCK". The sell is
            # the one spelled NASDAQ, so spelling order puts it first.
            dict(common, execution_uid="QT:5:b1", security_type="STOCK", side="BUY",
                 quantity=10, price=100.0, timestamp="2026-08-05T09:00:00-07:00"),
            dict(common, execution_uid="QT:5:b2", security_type="NASDAQ", side="SELL",
                 quantity=10, price=110.0, timestamp="2026-08-05T10:00:00-07:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        row = dict(conn.execute("SELECT * FROM trades").fetchone())
        legs = [dict(leg) for leg in conn.execute("SELECT * FROM trade_legs ORDER BY leg_id")]
    assert row["direction"] == "LONG", "the buy opened the position, not the sell"
    assert [leg["role"] for leg in legs] == ["OPEN", "CLOSE"]
    assert row["gross_pnl"] == pytest.approx(100.0)


def test_mixed_option_symbol_spellings_receive_fills_in_time_order(tmp_path):
    """Raw symbol spelling must not sort a normalized option group."""
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    common = dict(
        broker="IBKR", account_number="U1", account_label="U1", account_type="",
        security_type="OPT", currency="USD", trade_date="2026-08-05",
        commission=0.0, fees=0.0, gross_amount=None, net_amount=None,
        order_id="", exchange_exec_id="", raw_json="{}",
    )
    store.upsert_executions(
        [
            dict(common, execution_uid="IBKR:U1:o1", symbol="SPY260116C00500000",
                 side="BUY", quantity=1, price=5.0,
                 timestamp="2026-08-05T09:00:00-07:00"),
            # Spaces make this raw spelling sort before the compact spelling,
            # despite being the later fill. Both normalize to the same symbol.
            dict(common, execution_uid="IBKR:U1:o2", symbol="SPY   260116C00500000",
                 side="SELL", quantity=1, price=7.0,
                 timestamp="2026-08-05T10:00:00-07:00"),
        ]
    )

    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        trade = dict(conn.execute("SELECT * FROM trades").fetchone())
        roles = [row[0] for row in conn.execute("SELECT role FROM trade_legs ORDER BY leg_id")]
    assert trade["direction"] == "LONG"
    assert trade["gross_pnl"] == pytest.approx(200.0)
    assert roles == ["OPEN", "CLOSE"]


def test_an_option_multiplier_survives_the_brokers_spelling(tmp_path):
    """The store's multiplier rule read security_type verbatim and missed "Option"."""
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    common = dict(
        broker="QUESTRADE", account_number="5", account_label="M", account_type="",
        symbol="AAPL260116C00150000", currency="USD", trade_date="2026-08-05",
        commission=0.0, fees=0.0, gross_amount=None, net_amount=None, order_id="",
        exchange_exec_id="", raw_json="{}",
    )
    store.upsert_executions(
        [
            dict(common, execution_uid="QT:5:o1", security_type="Option", side="BUY",
                 quantity=1, price=5.0, timestamp="2026-08-05T09:00:00-07:00"),
            dict(common, execution_uid="QT:5:o2", security_type="Option", side="SELL",
                 quantity=1, price=7.0, timestamp="2026-08-05T10:00:00-07:00"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        gross = conn.execute("SELECT gross_pnl FROM trades").fetchone()[0]
    assert gross == pytest.approx(200.0), "one contract, $2.00 move, x100 - not $2.00"


def test_questrade_partial_fills_without_execution_ids_do_not_collapse_on_order_id(tmp_path):
    importer = QuestradeImporter.__new__(QuestradeImporter)
    account = {"number": "51830546", "type": "TFSA"}
    base = {
        "orderId": 4242, "symbol": "AAPL", "securityType": "Stock", "side": "Buy",
        "quantity": 5, "price": 150.0, "commission": 2.5, "currency": "USD",
    }
    first = importer.normalize_execution(
        {**base, "timestamp": "2026-08-05T09:31:00-07:00"}, account
    )
    second = importer.normalize_execution(
        {**base, "timestamp": "2026-08-05T09:31:01-07:00"}, account
    )

    assert first.execution_uid != second.execution_uid
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    store.upsert_executions([first, second])
    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_executions").fetchone()[0] == 2


def test_questrade_idless_fill_uid_ignores_payload_and_fee_drift():
    importer = QuestradeImporter.__new__(QuestradeImporter)
    account = {"number": "51830546", "type": "TFSA"}
    stable = {
        "orderId": 4242, "symbol": "AAPL", "securityType": "Stock", "side": "Buy",
        "quantity": 5, "price": 150.0, "timestamp": "2026-08-05T09:31:00-07:00",
        "currency": "USD",
    }
    original = importer.normalize_execution(
        {**stable, "commission": 2.5, "fees": 0.01, "brokerVersion": "old"}, account
    )
    amended = importer.normalize_execution(
        {**stable, "commission": 2.75, "fees": 0.02, "brokerVersion": "new", "newField": True},
        account,
    )

    assert original.execution_uid == amended.execution_uid
    assert original.raw_json != amended.raw_json
