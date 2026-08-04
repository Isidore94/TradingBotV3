"""The capture-side IB adapter (plan sec 5.3, Phase 3b).

This is the only warehouse module that opens a socket, and it is deliberately
thin: request construction, bar parsing, and error classification are pure
functions with offline tests, while the ``ibapi`` client itself is built lazily
and exercised only against a live TWS (``@pytest.mark.broker``).

Separation from the champions is absolute:

* capture connects on its **own** client id - 1010 streaming, 1011 nightly
  backfill - asserted at connect. 1003 is retired because it collided with the
  M1 dual scheduler and produced a silent Yahoo fallback (risk R2).
* every request passes through the shared pacer, so capture yields to champion
  traffic and to IB error 162/366;
* an error here is reported to the pacer tagged ``capture=True`` and therefore
  can never reach `_IBKR_HISTORICAL_FAILURE_COUNT` in the champion fetch
  boundary (risk R1).

Capture asks IB for ``formatDate=2`` (epoch seconds, UTC). The champion path
uses ``formatDate=1`` and reads the naive local strings that come back; for a
new, separate connection there is no reason to inherit that ambiguity, and an
unambiguous UTC instant is what the point-in-time contract wants. The tee is
unaffected: it reads bars the champion already parsed, in the champion's own
convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

try:  # package import
    from . import pacer as pacer_mod
    from .backfill import FetchResult
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import pacer as pacer_mod  # type: ignore
    from backfill import FetchResult  # type: ignore

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7496
EXCHANGE_TZ_NAME = "US/Eastern"

BAR_SIZES = {
    "M1": "1 min",
    "M5": "5 mins",
    "M15": "15 mins",
    "M30": "30 mins",
    "H1": "1 hour",
    "D1": "1 day",
}
# ETH runs 04:00-20:00 ET; a one-session request ends at the ETH close so a
# useRTH=0 request returns premarket and postmarket bars too (LD-03).
ETH_CLOSE_HOUR = 20
RTH_CLOSE_HOUR = 16


def historical_request(symbol: str, day: date, *, timeframe: str = "M5", use_rth: bool = False, duration: str = "1 D") -> dict:
    """The exact ``reqHistoricalData`` arguments for one (symbol, session).

    Pure: no connection, no state. Tested offline so a wrong duration or a
    dropped ``useRTH`` cannot slip through unnoticed.
    """
    bar_size = BAR_SIZES.get(str(timeframe).upper())
    if bar_size is None:
        raise ValueError(f"unsupported capture timeframe {timeframe!r}; known: {sorted(BAR_SIZES)}")
    close_hour = RTH_CLOSE_HOUR if use_rth else ETH_CLOSE_HOUR
    return {
        "symbol": str(symbol).strip().upper(),
        "endDateTime": f"{day:%Y%m%d} {close_hour:02d}:00:00 {EXCHANGE_TZ_NAME}",
        "durationStr": duration,
        "barSizeSetting": bar_size,
        "whatToShow": "TRADES",
        "useRTH": 1 if use_rth else 0,
        # Epoch seconds (UTC): no timezone inference anywhere downstream.
        "formatDate": 2,
        "keepUpToDate": False,
    }


def parse_bar(raw, *, interval: timedelta = timedelta(minutes=5)) -> dict | None:
    """One IB bar -> a normalized dict with a timezone-aware UTC start.

    A timestamp that cannot be read as an instant is dropped, never guessed:
    missing data is uncertainty (plan.md sec 5).
    """
    stamp = raw.get("date") if isinstance(raw, dict) else getattr(raw, "date", None)
    start = _epoch_to_utc(stamp)
    if start is None:
        return None

    def _get(name, *aliases):
        for key in (name, *aliases):
            if isinstance(raw, dict):
                if key in raw:
                    return raw[key]
            elif hasattr(raw, key):
                return getattr(raw, key)
        return None

    try:
        values = {
            "open": float(_get("open")),
            "high": float(_get("high")),
            "low": float(_get("low")),
            "close": float(_get("close")),
        }
    except (TypeError, ValueError):
        return None
    volume = _get("volume")
    wap = _get("wap", "vwap", "average")
    count = _get("barCount", "trade_count")
    return {
        "interval_start": start,
        "interval_end": start + interval,
        **values,
        "volume": _safe_float(volume) or 0.0,
        "vwap": _safe_float(wap),
        "trade_count": _safe_int(count),
    }


def _epoch_to_utc(value):
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:  # formatDate=2 gives epoch seconds
        return datetime.fromtimestamp(int(text), tz=timezone.utc)
    except (TypeError, ValueError):
        pass
    for fmt in ("%Y%m%d  %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class CaptureConnectionSpec:
    role: str
    client_id: int
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    def validate(self) -> int:
        """Assert the allocation before a socket is ever opened (R2)."""
        return pacer_mod.assert_client_id(self.client_id, self.role)


def backfill_connection_spec(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> CaptureConnectionSpec:
    spec = CaptureConnectionSpec(
        role=pacer_mod.ROLE_NIGHTLY_BACKFILL,
        client_id=pacer_mod.CLIENT_ID_NIGHTLY_BACKFILL,
        host=host,
        port=port,
    )
    spec.validate()
    return spec


def streaming_connection_spec(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> CaptureConnectionSpec:
    spec = CaptureConnectionSpec(
        role=pacer_mod.ROLE_CAPTURE_STREAM,
        client_id=pacer_mod.CLIENT_ID_CAPTURE_STREAMER,
        host=host,
        port=port,
    )
    spec.validate()
    return spec


class IbCaptureFetcher:
    """A backfill ``fetcher`` backed by an IB transport.

    ``transport.request_historical(**kwargs)`` returns ``(bars, error)`` where
    ``error`` is ``(code, message)`` or ``None``. Keeping the transport
    injectable is what makes the retry, classification, and parsing logic
    testable without a broker; :func:`build_ib_transport` supplies the real one.
    """

    def __init__(self, transport, *, spec: CaptureConnectionSpec, pacer=None, interval_by_timeframe=None):
        self.transport = transport
        self.spec = spec
        self.spec.validate()
        self.pacer = pacer or pacer_mod.get_pacer()
        self.intervals = interval_by_timeframe or {
            "M1": timedelta(minutes=1),
            "M5": timedelta(minutes=5),
            "M15": timedelta(minutes=15),
            "M30": timedelta(minutes=30),
            "H1": timedelta(hours=1),
        }

    def is_connected(self) -> bool:
        checker = getattr(self.transport, "is_connected", None)
        return bool(checker()) if callable(checker) else True

    def ensure_connected(self) -> bool:
        """Reconnect after the ~23:45 ET TWS restart; never raise for it."""
        if self.is_connected():
            return True
        # The real client exposes connect_spec(spec); ibapi's inherited
        # EClient.connect takes (host, port, clientId) and must not be handed a
        # spec object - preferring connect_spec is what makes the post-restart
        # reconnect actually work against the real transport.
        connect = getattr(self.transport, "connect_spec", None) or getattr(self.transport, "connect", None)
        if not callable(connect):
            return False
        try:
            connect(self.spec)
        except Exception:
            return False
        return self.is_connected()

    def __call__(self, symbol: str, day: date, *, timeframe: str = "M5", use_rth: bool = False) -> FetchResult:
        if not self.ensure_connected():
            return FetchResult(error_message="capture connection unavailable")
        request = historical_request(symbol, day, timeframe=timeframe, use_rth=use_rth)
        try:
            bars, error = self.transport.request_historical(**request)
        except Exception as exc:
            self.pacer.note_error(0, str(exc), capture=True)
            return FetchResult(error_message=str(exc))
        if error:
            code, message = (error if isinstance(error, (tuple, list)) else (0, str(error)))
            # capture=True is the isolation tag: this never reaches the
            # champion's Yahoo-only circuit breaker.
            self.pacer.note_error(code, message, capture=True)
            return FetchResult(error_code=int(code or 0), error_message=str(message or ""))
        interval = self.intervals.get(str(timeframe).upper(), timedelta(minutes=5))
        parsed = [bar for bar in (parse_bar(raw, interval=interval) for raw in bars or []) if bar]
        if parsed:
            self.pacer.note_capture_success()
        return FetchResult(bars=parsed)


def build_ib_transport(spec: CaptureConnectionSpec):  # pragma: no cover - requires TWS
    """The real ibapi transport. Imported lazily; verified live, not offline.

    Deliberately unexercised by the offline suite: a fake transport proves the
    logic above, and only a broker-marked run against TWS can prove the socket
    behaviour. Until that run happens on the desk, treat this function as
    unverified.
    """
    spec.validate()
    import threading

    from ibapi.client import EClient
    from ibapi.contract import Contract
    from ibapi.wrapper import EWrapper

    class _CaptureClient(EWrapper, EClient):
        def __init__(self):
            EWrapper.__init__(self)
            EClient.__init__(self, self)
            self._lock = threading.Lock()
            self._bars: dict[int, list] = {}
            self._errors: dict[int, tuple] = {}
            self._done: dict[int, threading.Event] = {}
            self._next_id = 1

        # -- ibapi callbacks
        def historicalData(self, reqId, bar):  # noqa: N802 (ibapi naming)
            with self._lock:
                self._bars.setdefault(reqId, []).append(bar)

        def historicalDataEnd(self, reqId, start, end):  # noqa: N802
            event = self._done.get(reqId)
            if event is not None:
                event.set()

        def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):  # noqa: N802
            if reqId is not None and reqId > 0:
                with self._lock:
                    self._errors[reqId] = (errorCode, errorString)
                event = self._done.get(reqId)
                if event is not None:
                    event.set()

        # -- transport API
        def is_connected(self) -> bool:
            return bool(self.isConnected())

        def connect_spec(self, connection: CaptureConnectionSpec):
            connection.validate()
            self.connect(connection.host, connection.port, connection.client_id)
            threading.Thread(target=self.run, daemon=True, name="warehouse-capture-ib").start()

        def request_historical(self, *, symbol, timeout=30.0, **kwargs):
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            with self._lock:
                req_id = self._next_id
                self._next_id += 1
                self._done[req_id] = threading.Event()
                self._bars[req_id] = []
            self.reqHistoricalData(reqId=req_id, contract=contract, chartOptions=[], **kwargs)
            self._done[req_id].wait(timeout=timeout)
            with self._lock:
                bars = self._bars.pop(req_id, [])
                error = self._errors.pop(req_id, None)
                self._done.pop(req_id, None)
            return bars, error

    client = _CaptureClient()
    client.connect_spec(spec)
    return client


__all__ = [
    "BAR_SIZES",
    "CaptureConnectionSpec",
    "ETH_CLOSE_HOUR",
    "IbCaptureFetcher",
    "RTH_CLOSE_HOUR",
    "backfill_connection_spec",
    "build_ib_transport",
    "historical_request",
    "parse_bar",
    "streaming_connection_spec",
]
