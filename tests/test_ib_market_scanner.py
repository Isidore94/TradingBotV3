"""ib_market_scanner: scannerData/scannerDataEnd parsing and connection handling, no network."""

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ib_market_scanner as ims  # noqa: E402


def _details(symbol):
    return SimpleNamespace(contract=SimpleNamespace(symbol=symbol, secType="STK"))


class FakeApp(ims._ScannerApp):
    """The real callbacks; the socket methods answer from a script."""

    def __init__(self, answers, *, in_use=False, log=None):
        super().__init__()
        self.answers = answers
        self.in_use = in_use
        self.log = log if log is not None else []
        self.connected = False
        self.cancelled = []
        self.subs = []

    def connect(self, host, port, clientId):  # noqa: N803
        self.log.append(("connect", host, port, clientId))
        self.connected = True

    def isConnected(self):  # noqa: N802
        return self.connected

    def run(self):
        if self.in_use:
            self.error(-1, 326, "Unable to connect as the client id is already in use.")
            return
        self.nextValidId(1)
        while self.connected:  # the real reader loop lives until the socket closes
            time.sleep(0.005)

    def disconnect(self):
        self.connected = False

    def reqScannerSubscription(self, reqId, subscription, options, filters):  # noqa: N802,N803
        self.subs.append(subscription)
        answer = self.answers.get(subscription.scanCode)
        if isinstance(answer, tuple):  # (code, text) error
            self.error(reqId, answer[0], answer[1])
            return
        if answer is None:  # never answers: timeout
            return
        # Delivered out of rank order, with a repeat, like a re-sent row.
        for rank, symbol in reversed(list(enumerate(answer))):
            self.scannerData(reqId, rank, _details(symbol), "", "", "", "")
        if answer:
            self.scannerData(reqId, 0, _details(answer[0]), "", "", "", "")
        self.scannerDataEnd(reqId)

    def cancelScannerSubscription(self, reqId):  # noqa: N802,N803
        self.cancelled.append(reqId)


def _scanner(apps, **kwargs):
    queue = list(apps)
    return ims.IBMarketScanner(host="127.0.0.1", port=7496, client_id=9135,
                               app_factory=lambda: queue.pop(0), connect_timeout_s=0.2, **kwargs)


def test_scanner_rows_come_back_in_rank_order_per_scan_and_are_cancelled():
    app = FakeApp({"TOP_PERC_GAIN": ["AAA", "BRK B", "CCC"], "TOP_PERC_LOSE": ["ZZZ"],
                   "HOT_BY_VOLUME": []})
    scanner = _scanner([app])
    result = scanner.run_scans()
    assert result == {"TOP_PERC_GAIN": ["AAA", "BRK-B", "CCC"], "TOP_PERC_LOSE": ["ZZZ"],
                      "HOT_BY_VOLUME": []}
    assert [s.scanCode for s in app.subs] == list(ims.SCAN_CODES)
    sub = app.subs[0]
    assert (sub.instrument, sub.locationCode, sub.numberOfRows) == ("STK", "STK.US.MAJOR", 50)
    assert sub.abovePrice == ims.ABOVE_PRICE and sub.aboveVolume == ims.ABOVE_VOLUME
    assert app.cancelled == [1, 2, 3]
    assert ims.pooled_symbols(result) == ["AAA", "BRK-B", "CCC", "ZZZ"]


def test_one_connection_is_kept_across_calls():
    log = []
    app = FakeApp({"TOP_PERC_GAIN": ["AAA"]}, log=log)
    scanner = _scanner([app])
    scanner.run_scans(["TOP_PERC_GAIN"])
    scanner.run_scans(["TOP_PERC_GAIN"])
    assert [entry[0] for entry in log] == ["connect"]


def test_dropped_connection_is_rebuilt():
    first = FakeApp({"TOP_PERC_GAIN": ["AAA"]})
    second = FakeApp({"TOP_PERC_GAIN": ["BBB"]})
    scanner = _scanner([first, second])
    assert scanner.run_scans(["TOP_PERC_GAIN"]) == {"TOP_PERC_GAIN": ["AAA"]}
    first.connected = False
    assert scanner.run_scans(["TOP_PERC_GAIN"]) == {"TOP_PERC_GAIN": ["BBB"]}


def test_client_id_in_use_retries_on_the_next_id():
    log = []
    busy = FakeApp({}, in_use=True, log=log)
    free = FakeApp({"TOP_PERC_GAIN": ["AAA"]}, log=log)
    scanner = _scanner([busy, free])
    assert scanner.run_scans(["TOP_PERC_GAIN"]) == {"TOP_PERC_GAIN": ["AAA"]}
    assert [entry[3] for entry in log] == [9135, 9136]
    assert scanner.client_id == 9136


def test_a_failed_scan_is_reported_and_the_rest_still_return():
    app = FakeApp({"TOP_PERC_GAIN": ["AAA"], "TOP_PERC_LOSE": (162, "no scanner permission")})
    scanner = _scanner([app])
    result = scanner.run_scans(["TOP_PERC_GAIN", "TOP_PERC_LOSE", "HOT_BY_VOLUME"], timeout_s=0.1)
    assert result == {"TOP_PERC_GAIN": ["AAA"]}
    assert "162" in scanner.last_errors["TOP_PERC_LOSE"]
    assert "timed out" in scanner.last_errors["HOT_BY_VOLUME"]


def test_every_scan_failing_raises_with_the_ib_error():
    app = FakeApp({"TOP_PERC_GAIN": (162, "Scanner subscription not allowed")})
    scanner = _scanner([app])
    with pytest.raises(ims.ScannerError, match="Scanner subscription not allowed"):
        scanner.run_scans(["TOP_PERC_GAIN"])


def test_refused_connection_raises_scanner_error():
    class Refused(FakeApp):
        def connect(self, host, port, clientId):  # noqa: N803
            self.error(-1, 502, "Couldn't connect to TWS.")

    scanner = _scanner([Refused({})])
    with pytest.raises(ims.ScannerError, match="502"):
        scanner.run_scans()


def test_scanner_module_never_places_orders_or_asks_for_data():
    tree = ast.parse((SCRIPTS_DIR / "ib_market_scanner.py").read_text(encoding="utf-8"))
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    forbidden = {"placeOrder", "reqHistoricalData", "reqMktData", "reqRealTimeBars",
                 "reqTickByTickData", "reqIds", "cancelOrder"}
    assert not (called & forbidden)
