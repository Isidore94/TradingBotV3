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
    """End to end on the live CVNA spread: socket rows by day, Flex that night.
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


# ---------------------------------------------------------------------------
# 3. A trade with a made-up entry is kept, flagged, and not counted
# ---------------------------------------------------------------------------


def _qt_row(uid, side, quantity, price, stamp, *, symbol="SMH", account="51830546", label="TFSA"):
    return {
        "execution_uid": uid, "broker": "QUESTRADE", "account_number": account,
        "account_label": label, "account_type": label, "symbol": symbol,
        "security_type": "UNKNOWN", "currency": "USD", "side": side, "quantity": quantity,
        "price": price, "timestamp": stamp, "trade_date": stamp[:10],
        "commission": 0.0, "fees": 0.0, "gross_amount": None, "net_amount": None,
        "order_id": "", "exchange_exec_id": "", "raw_json": "{}", "source": "QT_API",
    }


def _store_with_a_missing_buy(tmp_path):
    """The live SMH shape: 8 bought, 11 sold (3 buys never imported), 3 covered.
    Plus one honest AAPL round trip in the margin account."""
    from journal_store import JournalStore

    store = JournalStore(tmp_path / "journal.sqlite3")
    store.initialize_schema()
    store.upsert_executions([
        _qt_row("QT:51830546:1", "BUY", 8, 580.0, "2026-08-12T10:00:00-04:00"),
        _qt_row("QT:51830546:2", "SELL", 11, 589.49, "2026-08-13T15:46:08-04:00"),
        _qt_row("QT:51830546:3", "BUY", 3, 552.25, "2026-09-03T13:45:41-04:00"),
        _qt_row("QT:29347316:4", "BUY", 10, 100.0, "2026-08-12T10:00:00-04:00",
                symbol="AAPL", account="29347316", label="Margin"),
        _qt_row("QT:29347316:5", "SELL", 10, 110.0, "2026-08-13T10:00:00-04:00",
                symbol="AAPL", account="29347316", label="Margin"),
    ])
    store.set_account_tax_status("QUESTRADE", "51830546", "TAX_FREE")
    store.set_account_tax_status("QUESTRADE", "29347316", "TAXABLE")
    store.rebuild_trades(refresh_tags=False)
    return store


def test_a_trade_with_a_made_up_entry_is_kept_and_marked(tmp_path):
    store = _store_with_a_missing_buy(tmp_path)
    trades = {(t["symbol"], t["direction"]): t for t in store.list_trades()}

    fabricated = trades[("SMH", "SHORT")]
    assert fabricated["status"] == "CLOSED"
    assert fabricated["entry_invented"]
    assert not trades[("SMH", "LONG")]["entry_invented"]
    assert not trades[("AAPL", "LONG")]["entry_invented"]


def test_a_made_up_entry_is_not_in_the_totals_calendar_or_stats(tmp_path):
    from journal_analytics import build_analytics_summary, calendar_pnl_by_day, resolve_pnl_key

    store = _store_with_a_missing_buy(tmp_path)
    trades = store.list_trades()
    fabricated = next(t for t in trades if t["entry_invented"])
    honest = [t for t in trades if t["status"] == "CLOSED" and not t["entry_invented"]]

    summary = build_analytics_summary(trades, "Native")
    assert summary["overall"]["net_pnl"] == sum(t["net_pnl"] for t in honest)
    assert summary["overall"]["closed"] == len(honest)
    assert summary["not_counted"]["trades"] == 1
    assert summary["not_counted"]["net_pnl"] == fabricated["net_pnl"]
    assert "1 trade needs missing fills - not counted ($" in summary["not_counted"]["line"]

    by_day = calendar_pnl_by_day(trades)
    assert fabricated["closed_at"][:10] not in by_day

    key, _note = resolve_pnl_key([fabricated], "Native")
    assert key == "net_pnl"


def test_a_stock_short_in_a_registered_account_is_a_missing_buy(tmp_path):
    """A TFSA cannot short stock. A SHORT there means the buy was never imported,
    even when no SYNTHETIC_OPEN leg says so (the sell had nothing to close)."""
    from journal_analytics import has_invented_entry, is_registered_stock_short

    store = _store_with_a_missing_buy(tmp_path)
    store.upsert_executions([
        _qt_row("QT:51830546:9", "SELL", 5, 60.0, "2026-08-20T10:00:00-04:00", symbol="DRAM"),
    ])
    store.rebuild_trades(refresh_tags=False)
    dram = next(t for t in store.list_trades() if t["symbol"] == "DRAM")

    assert dram["direction"] == "SHORT"
    assert is_registered_stock_short(dram)
    assert has_invented_entry(dram)
    # A written option in a TFSA is allowed and is not flagged.
    sold_put = {**dram, "symbol": "AAOI18JUN26P120.00", "security_type": "UNKNOWN"}
    assert not is_registered_stock_short(sold_put)
    assert not is_registered_stock_short({**dram, "security_type": "OPT"})
    assert not is_registered_stock_short(
        {**dram, "account_tax_status": "TAXABLE", "account_label": "Margin"}
    )


def test_the_equity_curve_skips_a_made_up_entry(tmp_path, monkeypatch):
    from ui.services import journal_feed

    store = _store_with_a_missing_buy(tmp_path)
    monkeypatch.setattr(journal_feed, "_STORE", store)
    monkeypatch.setattr(journal_feed, "_store", lambda: store)
    trades = journal_feed.load_trades(date_from="2026-01-01", date_to="2026-12-31")
    points = journal_feed.equity_curve(trades, "Native")
    honest = sum(
        t.net_pnl for t in trades if t.is_closed and (t.symbol, t.direction) != ("SMH", "SHORT")
    )

    assert points[-1][1] == honest


def test_the_journal_tabs_say_what_they_left_out(tmp_path, monkeypatch):
    import pytest

    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ui.services import journal_feed

    QApplication.instance() or QApplication([])
    store = _store_with_a_missing_buy(tmp_path)
    monkeypatch.setattr(journal_feed, "_STORE", store)
    monkeypatch.setattr(journal_feed, "_store", lambda: store)
    from ui.panels.journal_panel import JournalPanel

    panel = JournalPanel()
    try:
        panel.header.range_input.setCurrentText("All")
        panel.analytics_tab.reload()
        assert "1 trade needs missing fills - not counted" in panel.analytics_tab.not_counted_note.text()
        assert panel.analytics_tab.not_counted_note.isVisibleTo(panel.analytics_tab)

        panel.calendar_tab.reload()
        assert "1 trade needs missing fills - not counted" in panel.calendar_tab.summary.text()
    finally:
        panel.shutdown()
        panel.deleteLater()


# ---------------------------------------------------------------------------
# 5. One list of side words
# ---------------------------------------------------------------------------


def test_every_reader_uses_the_one_side_list():
    import journal_file_authority
    import journal_reclassify
    import journal_store
    import journal_tax_report
    from journal_analytics import _normalize_side
    from journal_identity import BUY_SIDE_WORDS, PRE_TJ9Q_VERBATIM_SIDES, SELL_SIDE_WORDS
    from journal_importers import EXTENDED_SIDE_WORDS, normalize_side

    assert {"STO", "STC", "SELL", "SLD", "SHORT", "SSHORT", "SELLSHORT"} <= SELL_SIDE_WORDS
    assert {"BTO", "BTC", "BUY", "BOT", "COV", "COVER", "BUYTOCOVER"} <= BUY_SIDE_WORDS
    assert not SELL_SIDE_WORDS & BUY_SIDE_WORDS
    assert EXTENDED_SIDE_WORDS == PRE_TJ9Q_VERBATIM_SIDES
    assert journal_reclassify._BROKER_SELL_WORDS == SELL_SIDE_WORDS
    assert journal_file_authority._BUY_SIDES == BUY_SIDE_WORDS
    for word in SELL_SIDE_WORDS:
        assert normalize_side(word) == "SELL", word
        assert journal_tax_report._signed_quantity(word, 2.0) == -2.0, word
        assert _normalize_side(word) == "SHORT", word
    for word in BUY_SIDE_WORDS:
        assert normalize_side(word) == "BUY", word
        assert journal_tax_report._signed_quantity(word, 2.0) == 2.0, word
        assert _normalize_side(word) == "LONG", word
    # The rebuild reads every sell word as a sell except the ones the journal
    # still stores verbatim until the trader runs journal_reclassify (gate #162).
    for word in SELL_SIDE_WORDS - PRE_TJ9Q_VERBATIM_SIDES:
        assert journal_store._signed_quantity({"side": word, "quantity": 2}) == -2.0, word


def test_the_new_stat_cards_leave_out_a_made_up_entry():
    from journal_analytics import calendar_day_stats, trade_performance_stats

    real = {"trade_id": "R", "status": "CLOSED", "net_pnl": 10.0, "trade_date": "2026-06-10",
            "closed_at": "2026-06-10T15:00:00-04:00", "direction": "LONG"}
    made_up = {"trade_id": "M", "status": "CLOSED", "net_pnl": 400.0, "trade_date": "2026-06-10",
               "closed_at": "2026-06-10T15:05:00-04:00", "direction": "SHORT", "entry_invented": True}

    stats = trade_performance_stats([real, made_up], "net_pnl")
    assert stats["net"] == 10.0
    assert calendar_day_stats([real, made_up], pnl_key="net_pnl")["2026-06-10"]["net"] == 10.0
