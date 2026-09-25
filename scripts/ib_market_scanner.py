"""IB market scanner: today's top gainers, losers and volume names from TWS.

A standalone ibapi client on its own client id (default 9135; the journal
importer uses 9125-9127, the bot 1,250,000+, the D1/warehouse clients
1003-1011). It sends scanner subscriptions only: no orders, no historical or
market-data requests. One connection is kept across calls and rebuilt when it
drops. Built lazily on a worker thread, never on the Qt thread.

Probe once from a shell: ``python scripts/ib_market_scanner.py --probe``.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Callable, Iterable
from swallowed import note_swallowed

try:
    from ibapi.client import EClient
    from ibapi.scanner import ScannerSubscription
    from ibapi.wrapper import EWrapper

    IBAPI_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when ibapi is absent
    EClient = object  # type: ignore[assignment,misc]
    EWrapper = object  # type: ignore[assignment,misc]
    ScannerSubscription = None  # type: ignore[assignment]
    IBAPI_AVAILABLE = False

#: The scans Movers asks for, in this order.
SCAN_CODES = ("TOP_PERC_GAIN", "TOP_PERC_LOSE", "HOT_BY_VOLUME")
SCAN_ROWS = 50
SCAN_TIMEOUT_SECONDS = 15.0
ABOVE_PRICE = 5.0
ABOVE_VOLUME = 200_000
INSTRUMENT = "STK"
LOCATION_CODE = "STK.US.MAJOR"
#: Common stock only: drops ETFs and ETNs from the scans.
STOCK_TYPE_FILTER = "CORP"

CLIENT_ID_SETTING = "movers_ib_scanner_client_id"
DEFAULT_CLIENT_ID = 9135
CLIENT_ID_RETRY_COUNT = 3
#: Host/port: the scanner's own keys, then the journal importer's, then TWS live defaults.
HOST_SETTINGS = ("movers_ib_scanner_host", "journal_ibkr_host")
PORT_SETTINGS = ("movers_ib_scanner_port", "journal_ibkr_port")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7496
CONNECT_TIMEOUT_SECONDS = 5.0

CLIENT_ID_IN_USE = 326
#: Farm/connection status notices, not failures.
_INFO_CODES = {2103, 2104, 2105, 2106, 2107, 2108, 2119, 2157, 2158, 165}
_CONNECTION_LOST = {504, 1100, 1300, 502}


class ScannerError(RuntimeError):
    """No scanner answer: not connected, refused, or every scan failed."""


def _setting(keys: Iterable[str], default):
    try:
        from project_paths import get_local_setting
    except Exception:
        return default
    for key in keys:
        try:
            value = get_local_setting(key, None)
        except Exception:
            value = None
        if value not in (None, ""):
            return value
    return default


def resolve_connection() -> tuple[str, int, int]:
    """(host, port, client_id) from local settings, with safe defaults."""
    host = str(_setting(HOST_SETTINGS, DEFAULT_HOST) or DEFAULT_HOST)
    try:
        port = int(_setting(PORT_SETTINGS, DEFAULT_PORT))
    except (TypeError, ValueError):
        port = DEFAULT_PORT
    try:
        client_id = int(_setting((CLIENT_ID_SETTING,), DEFAULT_CLIENT_ID))
    except (TypeError, ValueError):
        client_id = DEFAULT_CLIENT_ID
    return host, port, client_id if client_id > 0 else DEFAULT_CLIENT_ID


#: IB suffixes for preferreds, warrants, units and rights ("DBRG PRJ", "XYZ WS").
_NON_COMMON_SUFFIXES = ("PR", "WS", "U", "RT", "W")


def is_common_symbol(symbol: str) -> bool:
    """False for an IB symbol whose class suffix marks a preferred, warrant, unit or right."""
    parts = str(symbol or "").strip().upper().split()
    return len(parts) < 2 or not any(parts[-1].startswith(s) for s in _NON_COMMON_SUFFIXES)


def ib_symbol_to_yahoo(symbol: str) -> str:
    """IB writes share classes with a space ("BRK B"); Yahoo and the lists use a dash."""
    return "-".join(str(symbol or "").strip().upper().split())


class _ScannerApp(EWrapper, EClient):  # type: ignore[misc]
    """The ibapi callbacks: collects ranked scanner rows per request id."""

    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.ready = threading.Event()
        self.client_id_in_use = False
        self.connection_errors: list[str] = []
        self._lock = threading.Lock()
        self._rows: dict[int, list[tuple[int, str]]] = {}
        self._done: dict[int, threading.Event] = {}
        self._errors: dict[int, str] = {}

    # --- request bookkeeping (called by IBMarketScanner) ---
    def expect(self, req_id: int) -> threading.Event:
        with self._lock:
            self._rows[req_id] = []
            self._errors.pop(req_id, None)
            event = self._done[req_id] = threading.Event()
        return event

    def take(self, req_id: int) -> tuple[list[str], str]:
        """Symbols in rank order (deduplicated) and the request's error text."""
        with self._lock:
            rows = self._rows.pop(req_id, [])
            self._done.pop(req_id, None)
            error = self._errors.pop(req_id, "")
        symbols: list[str] = []
        for _rank, symbol in sorted(rows, key=lambda row: row[0]):
            if symbol and symbol not in symbols:
                symbols.append(symbol)
        return symbols, error

    # --- ibapi callbacks ---
    def nextValidId(self, orderId: int) -> None:  # noqa: N802,N803 - ibapi callback name
        self.ready.set()

    def scannerData(self, reqId, rank, contractDetails, distance, benchmark, projection, legsStr=""):  # noqa: N802,N803
        contract = getattr(contractDetails, "contract", None)
        raw = getattr(contract, "symbol", "") or ""
        symbol = ib_symbol_to_yahoo(raw) if is_common_symbol(raw) else ""
        with self._lock:
            if reqId in self._rows and symbol:
                self._rows[reqId].append((int(rank), symbol))

    def scannerDataEnd(self, reqId: int) -> None:  # noqa: N802,N803
        with self._lock:
            event = self._done.get(reqId)
        if event is not None:
            event.set()

    def error(self, reqId, errorCode, errorString, *args: Any) -> None:  # noqa: N802,N803
        if errorCode in _INFO_CODES:
            return
        text = f"{errorCode}: {errorString}"
        if errorCode == CLIENT_ID_IN_USE:
            self.client_id_in_use = True
        with self._lock:
            event = self._done.get(reqId)
            if event is not None:
                self._errors[reqId] = text
        if event is not None:
            event.set()
            return
        self.connection_errors.append(text)
        if errorCode in _CONNECTION_LOST or errorCode == CLIENT_ID_IN_USE:
            self.ready.clear()

    def connectionClosed(self) -> None:  # noqa: N802
        self.ready.clear()


class IBMarketScanner:
    """One kept-alive scanner connection. Call from one worker thread at a time."""

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        app_factory: Callable[[], _ScannerApp] | None = None,
        connect_timeout_s: float = CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        default_host, default_port, default_client = resolve_connection()
        self.host = host or default_host
        self.port = int(port or default_port)
        self.base_client_id = int(client_id or default_client)
        self.client_id: int | None = None
        self._app_factory = app_factory or _ScannerApp
        self._connect_timeout = float(connect_timeout_s)
        self._app: _ScannerApp | None = None
        self._thread: threading.Thread | None = None
        self._next_req_id = 1
        self._lock = threading.Lock()
        self.last_errors: dict[str, str] = {}

    # --- connection ---
    def _alive(self) -> bool:
        app = self._app
        if app is None or not app.ready.is_set():
            return False
        try:
            return bool(app.isConnected()) and self._thread is not None and self._thread.is_alive()
        except Exception:
            return False

    def _connect(self) -> None:
        if not IBAPI_AVAILABLE and self._app_factory is _ScannerApp:
            raise ScannerError("ibapi is not installed")
        last = "no connection attempt"
        for offset in range(CLIENT_ID_RETRY_COUNT):
            client_id = self.base_client_id + offset
            app = self._app_factory()
            try:
                app.connect(self.host, self.port, client_id)
            except Exception as exc:
                raise ScannerError(f"connect {self.host}:{self.port} failed: {exc}") from exc
            if not app.isConnected():
                detail = "; ".join(app.connection_errors[-2:]) or "refused"
                raise ScannerError(f"connect {self.host}:{self.port} failed: {detail}")
            thread = threading.Thread(target=app.run, name="movers-ib-scanner", daemon=True)
            thread.start()
            if app.ready.wait(self._connect_timeout) and not app.client_id_in_use:
                self._app, self._thread, self.client_id = app, thread, client_id
                return
            last = "; ".join(app.connection_errors[-2:]) or "no nextValidId"
            self._drop(app, thread)
            if not app.client_id_in_use:
                break
        raise ScannerError(f"IB scanner client {self.base_client_id} not ready: {last}")

    @staticmethod
    def _drop(app, thread) -> None:
        try:
            app.disconnect()
        except Exception as exc:
            note_swallowed("IB scanner client disconnect failed", exc, quiet=True)
        if thread is not None:
            thread.join(timeout=2.0)

    def close(self) -> None:
        with self._lock:
            if self._app is not None:
                self._drop(self._app, self._thread)
            self._app = self._thread = None

    # --- scans ---
    def run_scans(
        self,
        scan_codes: Iterable[str] = SCAN_CODES,
        *,
        rows: int = SCAN_ROWS,
        timeout_s: float = SCAN_TIMEOUT_SECONDS,
        above_price: float = ABOVE_PRICE,
        above_volume: int = ABOVE_VOLUME,
    ) -> dict[str, list[str]]:
        """Symbols per scan code, in rank order. Raises ScannerError when no scan answered."""
        with self._lock:
            if not self._alive():
                if self._app is not None:
                    self._drop(self._app, self._thread)
                    self._app = self._thread = None
                self._connect()
            app = self._app
            assert app is not None
            results: dict[str, list[str]] = {}
            self.last_errors = {}
            for code in scan_codes:
                req_id = self._next_req_id
                self._next_req_id += 1
                sub = ScannerSubscription() if ScannerSubscription is not None else _PlainSubscription()
                sub.instrument = INSTRUMENT
                sub.locationCode = LOCATION_CODE
                sub.scanCode = str(code)
                sub.numberOfRows = int(rows)
                sub.abovePrice = float(above_price)
                sub.aboveVolume = int(above_volume)
                sub.stockTypeFilter = STOCK_TYPE_FILTER
                done = app.expect(req_id)
                try:
                    app.reqScannerSubscription(req_id, sub, [], [])
                    finished = done.wait(float(timeout_s))
                finally:
                    try:
                        app.cancelScannerSubscription(req_id)
                    except Exception as exc:
                        note_swallowed("IB scanner subscription cancel failed", exc, quiet=True)
                symbols, error = app.take(req_id)
                if error:
                    self.last_errors[code] = error
                elif not finished:
                    self.last_errors[code] = f"timed out after {timeout_s:g}s"
                else:
                    results[code] = symbols
            if not results:
                if not self._alive():
                    self._drop(app, self._thread)
                    self._app = self._thread = None
                detail = "; ".join(f"{k} {v}" for k, v in self.last_errors.items()) or "no scans"
                raise ScannerError(f"IB scanner returned nothing: {detail}")
            return results


class _PlainSubscription:
    """Attribute bag used only when ibapi is missing and a fake app is injected."""


def pooled_symbols(results: dict[str, list[str]]) -> list[str]:
    """All scan results as one list: order of the scans, then rank; no repeats."""
    return list(dict.fromkeys(s for names in results.values() for s in names))


def _probe() -> int:
    scanner = IBMarketScanner()
    print(f"IB scanner probe: {scanner.host}:{scanner.port} client {scanner.base_client_id}")
    try:
        results = scanner.run_scans()
    except ScannerError as exc:
        print(f"FAILED: {exc}")
        return 1
    finally:
        scanner.close()
    print(f"connected as client {scanner.client_id}")
    for code in SCAN_CODES:
        names = results.get(code)
        if names is None:
            print(f"{code}: ERROR {scanner.last_errors.get(code, '?')}")
        else:
            print(f"{code}: {len(names)} -> {', '.join(names[:10])}")
    print(f"pooled: {len(pooled_symbols(results))} names")
    return 0


if __name__ == "__main__":
    if "--probe" in sys.argv[1:]:
        logging.basicConfig(level=logging.WARNING)
        raise SystemExit(_probe())
    print("usage: python scripts/ib_market_scanner.py --probe")
    raise SystemExit(2)
