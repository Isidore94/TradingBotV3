"""Mentor P12: the IBKR book reader. A short-lived TWS client on its own id (9155): managed
accounts, positions, account summary, cancel, disconnect; nothing else is ever sent. A fake
EClient answers here; the suite never dials TWS."""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ibkr_positions as ip  # noqa: E402

pytest.importorskip("ibapi")
from ibapi.client import EClient  # noqa: E402
from ibapi.contract import Contract  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 9, 30, 7, 0, tzinfo=PT)
LIFECYCLE = {"connect", "isConnected", "run", "disconnect"}


def _contract(symbol, sec_type="STK", currency="USD", multiplier="", local=""):
    contract = Contract()
    contract.symbol, contract.secType, contract.currency = symbol, sec_type, currency
    contract.multiplier, contract.localSymbol = multiplier, local
    return contract


HAPPY_POSITIONS = [
    ("U111", _contract("NVDA"), 50, 120.5),
    ("U222", _contract("AMD"), -100, 150.0),
    ("U222", _contract("AAPL", "OPT", multiplier="100", local="AAPL  261016C00250000"), 2, 350.0),
    ("U222", _contract("TSLA"), 0, 200.0),  # flat today: not a position
]
HAPPY_SUMMARY = [
    ("U111", "TotalCashValue", "5000", "CAD"),
    ("U111", "CashBalance", "2500", "CAD"), ("U111", "CashBalance", "1200", "USD"),
    ("U111", "CashBalance", "3400", "BASE"), ("U111", "NetLiquidation", "15000", "CAD"),
    ("U222", "TotalCashValue", "-800", "USD"), ("U222", "NetLiquidation", "40000", "USD"),
]


class FakeReader(ip.IBKRPositionsReader):
    """TWS answers from lists; every EClient method name called is recorded."""

    def __init__(self, *, up=True, positions=HAPPY_POSITIONS, summary=HAPPY_SUMMARY, managed="U111,U222,",
                 answer_positions=True, answer_summary=True):
        super().__init__()
        self.calls: list[str] = []
        self.up, self.connected = up, False
        self._positions, self._summary, self._managed = positions, summary, managed
        self.answer_positions, self.answer_summary = answer_positions, answer_summary
        allowed = ip.ALLOWED_REQUESTS | LIFECYCLE
        for name in dir(EClient):
            if name.startswith("_") or name in allowed or not callable(getattr(EClient, name)):
                continue
            if name in ("managedAccounts", "position", "positionEnd", "accountSummary", "accountSummaryEnd",
                        "error", "connectionClosed"):
                continue
            setattr(self, name, self._forbidden(name))

    def _forbidden(self, name):
        def call(*args, **kwargs):
            self.calls.append(name)
            raise AssertionError(f"forbidden TWS call {name}")
        return call

    def connect(self, host, port, client_id):
        self.calls.append("connect")
        self.dialled = (host, port, client_id)
        if not self.up:
            raise ConnectionRefusedError("TWS down")
        self.connected = True

    def isConnected(self):  # noqa: N802
        self.calls.append("isConnected")
        return self.connected

    def run(self):
        self.calls.append("run")

    def disconnect(self):
        self.calls.append("disconnect")
        self.connected = False

    def reqManagedAccts(self):  # noqa: N802
        self.calls.append("reqManagedAccts")
        self.managedAccounts(self._managed)

    def reqPositions(self):  # noqa: N802
        self.calls.append("reqPositions")
        if self.answer_positions:
            for account, contract, qty, avg in self._positions:
                self.position(account, contract, qty, avg)
            self.positionEnd()

    def cancelPositions(self):  # noqa: N802
        self.calls.append("cancelPositions")

    def reqAccountSummary(self, req_id, group, tags):  # noqa: N802
        self.calls.append("reqAccountSummary")
        self.summary_args = (req_id, group, tags)
        if self.answer_summary:
            for row in self._summary:
                self.accountSummary(req_id, *row)
            self.accountSummaryEnd(req_id)

    def cancelAccountSummary(self, req_id):  # noqa: N802
        self.calls.append("cancelAccountSummary")


JOURNAL_ACCOUNTS = [
    {"broker": "IBKR", "account_number": "U111", "account_label": "IBKR TFSA", "account_type": "TFSA",
     "tax_status": "TAX_FREE"},
    {"broker": "QUESTRADE", "account_number": "111", "account_label": "TFSA 111", "account_type": "TFSA"},
]


def _fetch(reader, timeout=2.0):
    return reader.fetch("127.0.0.1", 7496, ip.IBKR_BOOK_CLIENT_ID, timeout, now=NOW, journal_accounts=JOURNAL_ACCOUNTS)


def test_the_fixed_client_id_is_9155_and_never_another_desk_id():
    assert ip.IBKR_BOOK_CLIENT_ID == 9155 and ip.IBKR_BOOK_CLIENT_ID not in (9125, 9135, 9145)
    assert ip.CLIENT_ID_SETTING == "mentor_ibkr_client_id"


def test_happy_path_two_accounts_and_an_option():
    reader = FakeReader()
    snap, reason = _fetch(reader)
    assert reason == "" and snap is not None and snap.broker == "IBKR"
    assert snap.fetched_utc == "2026-09-30T14:00:00+00:00"
    accounts = {a["account_number"]: a for a in snap.accounts}
    assert list(accounts) == ["U111", "U222"]
    assert accounts["U111"]["account_label"] == "IBKR TFSA" and accounts["U111"]["account_type"] == "TFSA"
    assert accounts["U111"]["cash"] == {"CAD": 2500.0, "USD": 1200.0}, "per currency, BASE dropped, no FX"
    assert accounts["U222"]["account_label"] == "U222", "no journal row: the account number"
    assert accounts["U222"]["account_type"] == "", "never assume margin"
    assert accounts["U222"]["cash"] == {"USD": -800.0} and accounts["U222"]["net_liquidation"] == {"USD": 40000.0}
    positions = {p["symbol"]: p for p in snap.positions}
    assert set(positions) == {"NVDA", "AMD", "AAPL 261016C00250000"}, "a flat row is not a position"
    assert positions["AMD"]["side"] == "SHORT" and positions["AMD"]["open_qty"] == 100.0
    assert positions["NVDA"]["avg_price"] == 120.5 and positions["NVDA"]["market_value"] is None
    option = positions["AAPL 261016C00250000"]
    assert option["security_type"] == "OPT" and option["multiplier"] == 100.0 and option["avg_price"] == 3.5
    assert all(p["broker"] == "IBKR" for p in snap.positions)
    assert reader.dialled == ("127.0.0.1", 7496, 9155)
    assert reader.summary_args == (ip.SUMMARY_REQ_ID, "All", ip.SUMMARY_TAGS)
    assert reader.calls[-3:] == ["cancelPositions", "cancelAccountSummary", "disconnect"]


def test_only_the_allowed_requests_are_ever_called():
    reader = FakeReader()
    _fetch(reader)
    assert set(reader.calls) <= ip.ALLOWED_REQUESTS | LIFECYCLE, reader.calls
    assert {"reqManagedAccts", "reqPositions", "reqAccountSummary"} <= set(reader.calls)
    source = (SCRIPTS_DIR / "ibkr_positions.py").read_text(encoding="utf-8")
    for word in ("placeOrder", "reqOpenOrders", "reqMktData", "reqAccountUpdates", "reqExecutions", "cancelOrder"):
        assert word not in source, word


def test_tws_down_is_none_with_a_reason_and_no_request():
    reader = FakeReader(up=False)
    snap, reason = _fetch(reader)
    assert snap is None and reason.startswith("TWS not running")
    assert not {"reqManagedAccts", "reqPositions", "reqAccountSummary"} & set(reader.calls)


def test_a_timeout_is_none_and_still_cancels_and_disconnects():
    reader = FakeReader(answer_positions=False)
    snap, reason = _fetch(reader, timeout=0.2)
    assert snap is None and reason.startswith("timeout: no positionEnd")
    assert reader.calls[-2:] == ["cancelPositions", "disconnect"] and "reqAccountSummary" not in reader.calls


def test_position_end_before_any_position_is_a_valid_empty_book():
    snap, reason = _fetch(FakeReader(positions=[]))
    assert reason == "" and snap is not None and snap.positions == ()
    assert [a["account_number"] for a in snap.accounts] == ["U111", "U222"]


def test_a_missing_summary_end_keeps_the_positions_with_cash_unknown():
    snap, reason = _fetch(FakeReader(answer_summary=False), timeout=0.3)
    assert reason == "" and len(snap.positions) == 3
    assert all(not a["cash_known"] for a in snap.accounts) and snap.errors


def test_fetch_book_never_raises_and_uses_the_setting_id(monkeypatch):
    import project_paths

    seen = {}
    monkeypatch.setattr(project_paths, "get_local_setting",
                        lambda key, default=None: 9161 if key == ip.CLIENT_ID_SETTING else default)

    def factory():
        reader = FakeReader()
        seen["reader"] = reader
        return reader

    snap, reason = ip.fetch_book(NOW, reader_factory=factory, journal_accounts=lambda: JOURNAL_ACCOUNTS)
    assert reason == "" and snap.broker == "IBKR" and seen["reader"].dialled[2] == 9161

    def broken():
        raise RuntimeError("boom")

    assert ip.fetch_book(NOW, reader_factory=broken, journal_accounts=lambda: []) == (
        None, "IBKR reader unavailable (RuntimeError: boom)")


def test_the_real_reader_against_the_suite_stub_is_tws_not_running():
    """The conftest stub refuses ``EClient.connect``: the real reader turns it into a reason."""
    snap, reason = ip.IBKRPositionsReader().fetch(timeout_sec=0.2, now=NOW)
    assert snap is None and reason.startswith("TWS not running")


def test_the_snapshot_round_trips_with_its_broker():
    snap, _ = _fetch(FakeReader())
    again = ip.BookSnapshot.from_json(snap.as_json())
    assert again == snap and again.broker == "IBKR"


# ---------------------------------------------------------------- review advisories (P12 round 1)
class _HangingConnect(FakeReader):
    """TWS accepts the socket and never answers: ibapi's connect would loop forever."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.release = threading.Event()

    def connect(self, host, port, client_id):
        self.calls.append("connect")
        self.connected = True
        self.release.wait(10)


def test_a_connect_that_never_returns_is_cut_at_the_deadline_and_disconnected():
    reader = _HangingConnect()
    started = time.monotonic()
    try:
        snap, reason = _fetch(reader, timeout=0.3)
        elapsed = time.monotonic() - started
    finally:
        reader.release.set()
    assert snap is None and reason.startswith("TWS not answering") and elapsed < 2.5
    assert reader.calls[-1] == "disconnect"
    assert not {"reqManagedAccts", "reqPositions", "reqAccountSummary"} & set(reader.calls)


class _RaisingPositions(FakeReader):
    def reqPositions(self):  # noqa: N802
        self.calls.append("reqPositions")
        raise RuntimeError("socket gone")


class _ClosingPositions(FakeReader):
    def reqPositions(self):  # noqa: N802
        self.calls.append("reqPositions")
        self.connected = False
        self.connectionClosed()


def test_disconnect_is_the_last_call_when_a_request_raises():
    reader = _RaisingPositions()
    snap, reason = _fetch(reader)
    assert snap is None and reason == "RuntimeError: socket gone"
    assert reader.calls[-2:] == ["cancelPositions", "disconnect"]


def test_disconnect_is_the_last_call_when_tws_closes_the_connection():
    reader = _ClosingPositions()
    snap, reason = _fetch(reader)
    assert snap is None and reason.startswith("TWS closed the connection")
    assert reader.calls[-1] == "disconnect" and "reqAccountSummary" not in reader.calls
