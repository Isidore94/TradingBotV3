"""Journal P&L correctness packet (2026-09-23).

Four input defects, each measured on a read-only copy of the trader's journal:
Flex times labelled with the desk's zone, socket/Flex duplicate fills, trades
with a made-up entry counted in totals, and side words that disagree.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from journal_importers import parse_ibkr_flex_statement  # noqa: E402

CVNA_FLEX_XML = """
<FlexQueryResponse><FlexStatements><FlexStatement accountId="U5102524"><Trades>
  <Trade accountId="U5102524" assetCategory="OPT" symbol="CVNA  260618P00063000"
         underlyingSymbol="CVNA" expiry="20260618" strike="63" putCall="P" multiplier="100"
         buySell="SELL" quantity="-1" tradePrice="1.51" dateTime="20260612;145118"
         ibExecID="00021ab9.6a2c031e.03.01" ibOrderID="393387315" ibCommission="-0.7046506"
         netCash="150.2953494" currency="USD"/>
  <Trade accountId="U5102524" assetCategory="OPT" symbol="CVNA  260618P00064500"
         underlyingSymbol="CVNA" expiry="20260618" strike="64.5" putCall="P" multiplier="100"
         buySell="BUY" quantity="1" tradePrice="2.17" dateTime="20260612;145118"
         ibExecID="00021ab9.6a2c031e.02.01" ibOrderID="393387315" ibCommission="-0.69825"
         netCash="-217.69825" currency="USD"/>
</Trades></FlexStatement></FlexStatements></FlexQueryResponse>
"""


# ---------------------------------------------------------------------------
# 1. Flex dateTime is Eastern
# ---------------------------------------------------------------------------


def test_a_flex_datetime_is_read_as_eastern_not_the_desk_zone():
    """The live CVNA spread filled at 11:51:18 Pacific (the socket says so).
    Flex writes that fill as 14:51:18 with no zone; it is New York time."""
    execution = parse_ibkr_flex_statement(CVNA_FLEX_XML)[0]
    stamp = datetime.fromisoformat(execution.timestamp)

    assert stamp.utcoffset() == timedelta(hours=-4)
    assert stamp == datetime(2026, 6, 12, 11, 51, 18, tzinfo=ZoneInfo("America/Vancouver"))
    assert execution.trade_date == "2026-06-12"


def test_a_flex_datetime_that_names_its_zone_keeps_it():
    xml = CVNA_FLEX_XML.replace('dateTime="20260612;145118"', 'dateTime="20260612;145118 US/Central"', 1)
    execution = parse_ibkr_flex_statement(xml)[0]
    stamp = datetime.fromisoformat(execution.timestamp)

    assert stamp.utcoffset() == timedelta(hours=-5)


# ---------------------------------------------------------------------------
# 2. A socket combo-leg fill and its Flex row are one execution
# ---------------------------------------------------------------------------


def _socket_importer():
    from journal_importers import IBKRExecutionImporter

    importer = IBKRExecutionImporter.__new__(IBKRExecutionImporter)
    importer.commissions = {}
    importer.quarantined = []
    return importer


def _socket_leg(exec_id: str, local_symbol: str, strike: float, side: str, price: float):
    from types import SimpleNamespace

    contract = SimpleNamespace(
        localSymbol=local_symbol, symbol="CVNA", secType="OPT",
        lastTradeDateOrContractMonth="20260618", strike=strike, right="P",
        currency="USD", exchange="CBOE2",
    )
    execution = SimpleNamespace(
        time="20260612  11:51:18", execId=exec_id, acctNumber="U5102524",
        side=side, shares=1.0, price=price, orderId="-982", permId="",
    )
    return contract, execution


def _socket_bag():
    from types import SimpleNamespace

    contract = SimpleNamespace(
        localSymbol="", symbol="CVNA", secType="BAG", lastTradeDateOrContractMonth="",
        strike=0.0, right="", currency="USD", exchange="SMART",
    )
    execution = SimpleNamespace(
        time="20260612  11:51:18", execId="0001640e.6a2b8893.01.01", acctNumber="U5102524",
        side="BOT", shares=1.0, price=0.66, orderId="-982", permId="",
    )
    return contract, execution


def test_a_socket_combo_leg_exec_id_lands_on_the_flex_uid():
    """Live: socket `...6a2c031e.03.01.01`, Flex `...6a2c031e.03.01` - one fill."""
    socket_row = _socket_importer().normalize_execution(
        *_socket_leg("00021ab9.6a2c031e.03.01.01", "CVNA  260618P00063000", 63.0, "SLD", 1.51)
    )
    flex_row = parse_ibkr_flex_statement(CVNA_FLEX_XML)[0]

    assert socket_row.execution_uid == flex_row.execution_uid
    assert datetime.fromisoformat(socket_row.timestamp) == datetime.fromisoformat(flex_row.timestamp)


def test_the_socket_import_drops_the_combo_parent_row():
    """IBKR reports a combo twice over the socket: the BAG parent and its legs.
    Flex reports only the legs. The parent row became a fake OPEN BAG trade."""
    importer = _socket_importer()
    importer.executions = [
        dict(zip(("contract", "execution"), _socket_bag())),
        dict(zip(("contract", "execution"), _socket_leg(
            "00021ab9.6a2c031e.02.01.01", "CVNA  260618P00064500", 64.5, "BOT", 2.17))),
    ]
    results = importer._normalized_results()

    assert [row.security_type for row in results] == ["OPT"]


def test_the_socket_then_flex_spread_rebuilds_one_position_per_leg(tmp_path):
    """End to end on the live CVNA spread: socket rows by day, Flex that night,
    One position per leg, each opened once: no BAG, no doubled quantity."""
    from journal_store import JournalStore

    store = JournalStore(tmp_path / "journal.sqlite3")
    store.initialize_schema()
    importer = _socket_importer()
    importer.executions = [
        dict(zip(("contract", "execution"), _socket_bag())),
        dict(zip(("contract", "execution"), _socket_leg(
            "00021ab9.6a2c031e.02.01.01", "CVNA  260618P00064500", 64.5, "BOT", 2.17))),
        dict(zip(("contract", "execution"), _socket_leg(
            "00021ab9.6a2c031e.03.01.01", "CVNA  260618P00063000", 63.0, "SLD", 1.51))),
    ]
    store.upsert_executions(importer._normalized_results())
    store.upsert_executions(parse_ibkr_flex_statement(CVNA_FLEX_XML))
    store.rebuild_trades(refresh_tags=False)

    with store.connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM raw_executions").fetchone()[0]
    trades = store.list_trades()
    assert count == 2
    assert sorted((t["symbol"], t["status"], t["quantity_opened"]) for t in trades) == [
        ("CVNA260618P00063000", "OPEN", 1.0),
        ("CVNA260618P00064500", "OPEN", 1.0),
    ]


def test_fills_in_different_zones_are_ordered_by_instant(tmp_path):
    """A Flex buy at 14:00 New York is BEFORE a socket sell at 11:30 Pacific.
    Sorting the text put the sell first and invented a short."""
    from journal_importers import NormalizedExecution
    from journal_store import JournalStore

    def row(uid, side, stamp, source):
        return NormalizedExecution(
            execution_uid=uid, source=source, broker="IBKR", account_number="U1",
            account_label="U1", account_type="", symbol="AAPL", security_type="STK",
            currency="USD", side=side, quantity=10.0, price=100.0, timestamp=stamp,
            trade_date=stamp[:10], commission=0.0, fees=0.0, gross_amount=None,
            net_amount=None, order_id="", exchange_exec_id=uid, raw_json="{}",
        )

    store = JournalStore(tmp_path / "journal.sqlite3")
    store.initialize_schema()
    store.upsert_executions([
        row("IBKR:U1:b", "SELL", "2026-06-12T11:30:00-07:00", "IBKR_SOCKET"),
        row("IBKR:U1:a", "BUY", "2026-06-12T14:00:00-04:00", "IBKR_FLEX"),
    ])
    store.rebuild_trades(refresh_tags=False)

    trades = store.list_trades()
    assert [(t["direction"], t["status"]) for t in trades] == [("LONG", "CLOSED")]


def test_only_the_exact_combo_leg_shape_is_trimmed():
    from journal_identity import canonical_ibkr_exec_id

    assert canonical_ibkr_exec_id("00021ab9.6a2c031e.03.01.01") == "00021ab9.6a2c031e.03.01"
    assert canonical_ibkr_exec_id("00021ab9.6a2c031e.03.01") == "00021ab9.6a2c031e.03.01"
    assert canonical_ibkr_exec_id("00021ab9.6a2c031e.03.01.02") == "00021ab9.6a2c031e.03.01.02"
    assert canonical_ibkr_exec_id("socket-1") == "socket-1"
