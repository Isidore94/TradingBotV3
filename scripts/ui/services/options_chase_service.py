"""Options chase: option chains and snapshot quotes for the Movers Pop names.

Runs on the Movers worker thread, after the tick's final board: for at most
OPTIONS_CHASE_MAX_PER_TICK Pop names with RVOL >= 2 (top by rank) it fetches the
option chain and snapshot quotes through ITS OWN IB client id (default 9145, the
`ib_market_scanner` pattern: one kept-alive connection, rebuilt when it drops),
caches them per symbol for OPTIONS_CHASE_CACHE_SECONDS, and asks the pure
`options_chase.pick_candidate` for the contract. Snapshot quotes only
(non-regulatory, never a streaming line), requests spaced by IB_REQUEST_GAP_SECONDS.
If IB is not connected, or the account has no option market-data permission,
it says so once per session and every Pop row reads "no option data (reason)";
a missing permission stops requests for the rest of the session.

Decision support only: no order is ever built or sent. Nothing here runs on the
Qt thread; the board only paints the `opt` result each Pop row carries. Every new
candidate and refusal is a row in `OPTIONS_CHASE_LOG_FILE`; a failed write loses
the rows, never the board.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import options_chase

try:
    from ibapi.client import EClient
    from ibapi.contract import Contract
    from ibapi.wrapper import EWrapper

    IBAPI_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when ibapi is absent
    EClient = object  # type: ignore[assignment,misc]
    EWrapper = object  # type: ignore[assignment,misc]
    Contract = None  # type: ignore[assignment]
    IBAPI_AVAILABLE = False

#: Pop names chased per Movers tick (RVOL >= 2, top by rank).
OPTIONS_CHASE_MAX_PER_TICK = 3
#: A symbol's chain and quotes are reused for this long; a quote this young is "fresh".
OPTIONS_CHASE_CACHE_SECONDS = 300
#: IB pacing: gap between two requests (IB allows 50 messages/s; this stays under 20).
IB_REQUEST_GAP_SECONDS = 0.05
#: Waits: contract details / option parameters, and one batch of snapshot quotes.
CHAIN_TIMEOUT_SECONDS = 8.0
QUOTE_TIMEOUT_SECONDS = 6.0
#: Live market data only (1); a delayed option quote is no use for a chase.
MARKET_DATA_TYPE_LIVE = 1

CLIENT_ID_SETTING = "options_chase_ib_client_id"
DEFAULT_CLIENT_ID = 9145
CLIENT_ID_RETRY_COUNT = 3
CONNECT_TIMEOUT_SECONDS = 5.0
CLIENT_ID_IN_USE = 326
#: IB codes that mean "no market-data permission" for the contract.
NO_PERMISSION_CODES = frozenset({354, 10089, 10090, 10091, 10168, 10186, 10197})
_INFO_CODES = {2103, 2104, 2105, 2106, 2107, 2108, 2119, 2157, 2158, 165, 10167}
_CONNECTION_LOST = {504, 1100, 1300, 502}
#: tickPrice types: bid, ask (live and delayed); tickOptionComputation: model (live, delayed).
_BID_TICKS = {1, 66}
_ASK_TICKS = {2, 67}
_MODEL_TICKS = {13, 83}


def note_swallowed(reason, exc=None, **kwargs):
    try:
        from swallowed import note_swallowed as _note
    except ImportError:
        return
    _note(reason, exc, **kwargs)


class OptionDataError(RuntimeError):
    """No option data for a name: not connected, no permission, or no chain."""

    def __init__(self, reason: str, *, permission: bool = False, connection: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.permission = permission
        self.connection = connection


def resolve_connection() -> tuple[str, int, int]:
    """(host, port, client_id): the scanner's host/port settings, this feature's own client id."""
    import ib_market_scanner as ims

    host = str(ims._setting(ims.HOST_SETTINGS, ims.DEFAULT_HOST) or ims.DEFAULT_HOST)
    try:
        port = int(ims._setting(ims.PORT_SETTINGS, ims.DEFAULT_PORT))
    except (TypeError, ValueError):
        port = ims.DEFAULT_PORT
    try:
        client_id = int(ims._setting((CLIENT_ID_SETTING,), DEFAULT_CLIENT_ID))
    except (TypeError, ValueError):
        client_id = DEFAULT_CLIENT_ID
    return host, port, client_id if client_id > 0 else DEFAULT_CLIENT_ID


# ------------------------------------------------------------------ IB client
class _OptionApp(EWrapper, EClient):  # type: ignore[misc]
    """The ibapi callbacks: contract ids, option parameters and snapshot quotes per request."""

    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.ready = threading.Event()
        self.client_id_in_use = False
        self.connection_errors: list[str] = []
        self._lock = threading.Lock()
        self._rows: dict[int, list[Any]] = {}
        self._quotes: dict[int, dict[str, float]] = {}
        self._done: dict[int, threading.Event] = {}
        self._errors: dict[int, list[int]] = {}

    def expect(self, req_id: int) -> threading.Event:
        with self._lock:
            self._rows[req_id] = []
            self._quotes[req_id] = {}
            self._errors[req_id] = []
            event = self._done[req_id] = threading.Event()
        return event

    def take(self, req_id: int) -> tuple[list[Any], dict[str, float], list[int]]:
        with self._lock:
            self._done.pop(req_id, None)
            return (self._rows.pop(req_id, []), self._quotes.pop(req_id, {}),
                    self._errors.pop(req_id, []))

    def _finish(self, req_id: int) -> None:
        with self._lock:
            event = self._done.get(req_id)
        if event is not None:
            event.set()

    # --- ibapi callbacks ---
    def nextValidId(self, orderId: int) -> None:  # noqa: N802,N803 - ibapi callback name
        self.ready.set()

    def contractDetails(self, reqId, contractDetails) -> None:  # noqa: N802,N803
        con_id = getattr(getattr(contractDetails, "contract", None), "conId", 0)
        with self._lock:
            if reqId in self._rows and con_id:
                self._rows[reqId].append(int(con_id))

    def contractDetailsEnd(self, reqId) -> None:  # noqa: N802,N803
        self._finish(reqId)

    def securityDefinitionOptionParameter(  # noqa: N802
        self, reqId, exchange, underlyingConId, tradingClass, multiplier, expirations, strikes  # noqa: N803
    ) -> None:
        with self._lock:
            if reqId in self._rows:
                self._rows[reqId].append({
                    "exchange": str(exchange or ""), "tradingClass": str(tradingClass or ""),
                    "multiplier": str(multiplier or "100"),
                    "expirations": sorted(expirations or ()), "strikes": sorted(strikes or ()),
                })

    def securityDefinitionOptionParameterEnd(self, reqId) -> None:  # noqa: N802,N803
        self._finish(reqId)

    def tickPrice(self, reqId, tickType, price, attrib) -> None:  # noqa: N802,N803
        if price is None or price < 0:
            return
        key = "bid" if tickType in _BID_TICKS else "ask" if tickType in _ASK_TICKS else ""
        if key:
            with self._lock:
                if reqId in self._quotes:
                    self._quotes[reqId][key] = float(price)

    def tickOptionComputation(  # noqa: N802
        self, reqId, tickType, tickAttrib, impliedVol, delta, optPrice,  # noqa: N803
        pvDividend, gamma, vega, theta, undPrice,  # noqa: N803
    ) -> None:
        if tickType not in _MODEL_TICKS:
            return
        with self._lock:
            quote = self._quotes.get(reqId)
            if quote is None:
                return
            if impliedVol is not None and 0 < impliedVol < 20:
                quote["iv"] = float(impliedVol)
            if delta is not None and -1.0 <= delta <= 1.0:
                quote["delta"] = float(delta)

    def tickSnapshotEnd(self, reqId) -> None:  # noqa: N802,N803
        self._finish(reqId)

    def error(self, reqId, errorCode, errorString, *args: Any) -> None:  # noqa: N802,N803
        if errorCode in _INFO_CODES:
            return
        if errorCode == CLIENT_ID_IN_USE:
            self.client_id_in_use = True
        with self._lock:
            known = reqId in self._errors
            if known:
                self._errors[reqId].append(int(errorCode))
        if known:
            self._finish(reqId)
            return
        self.connection_errors.append(f"{errorCode}: {errorString}")
        if errorCode in _CONNECTION_LOST or errorCode == CLIENT_ID_IN_USE:
            self.ready.clear()

    def connectionClosed(self) -> None:  # noqa: N802
        self.ready.clear()


def _stock_contract(symbol: str):
    contract = Contract() if Contract is not None else _Plain()
    contract.symbol = symbol.replace("-", " ")
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


def _option_contract(symbol: str, expiry: str, strike: float, right: str, params: Mapping[str, Any]):
    contract = Contract() if Contract is not None else _Plain()
    contract.symbol = symbol.replace("-", " ")
    contract.secType = "OPT"
    contract.exchange = "SMART"
    contract.currency = "USD"
    contract.lastTradeDateOrContractMonth = expiry.replace("-", "")
    contract.strike = float(strike)
    contract.right = right
    contract.multiplier = str(params.get("multiplier") or "100")
    if params.get("tradingClass"):
        contract.tradingClass = str(params["tradingClass"])
    return contract


class _Plain:
    """Attribute bag used only when ibapi is missing and a fake app is injected."""


def select_params(rows: Iterable[Mapping[str, Any]], symbol: str) -> dict[str, Any]:
    """The SMART, same-trading-class parameter row with the most expiries."""
    key = symbol.replace("-", " ").upper()
    ranked = sorted(
        (r for r in rows if r.get("expirations") and r.get("strikes")),
        key=lambda r: (str(r.get("exchange") or "").upper() != "SMART",
                       str(r.get("tradingClass") or "").upper() != key,
                       -len(r.get("expirations") or ())),
    )
    return dict(ranked[0]) if ranked else {}


class IBOptionChainClient:
    """One kept-alive option-data connection on its own client id. One worker thread at a time."""

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        app_factory: Callable[[], _OptionApp] | None = None,
        connect_timeout_s: float = CONNECT_TIMEOUT_SECONDS,
        chain_timeout_s: float = CHAIN_TIMEOUT_SECONDS,
        quote_timeout_s: float = QUOTE_TIMEOUT_SECONDS,
        request_gap_s: float = IB_REQUEST_GAP_SECONDS,
    ) -> None:
        if host is None or port is None or client_id is None:
            default_host, default_port, default_client = resolve_connection()
        else:
            default_host, default_port, default_client = host, port, client_id
        self.host = host or default_host
        self.port = int(port or default_port)
        self.base_client_id = int(client_id or default_client)
        self.client_id: int | None = None
        self._app_factory = app_factory or _OptionApp
        self._connect_timeout = float(connect_timeout_s)
        self._chain_timeout = float(chain_timeout_s)
        self._quote_timeout = float(quote_timeout_s)
        self._gap = float(request_gap_s)
        self._app: _OptionApp | None = None
        self._thread: threading.Thread | None = None
        self._next_req_id = 1
        self._lock = threading.Lock()
        self._params: dict[str, tuple[date, dict[str, Any]]] = {}

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
        if not IBAPI_AVAILABLE and self._app_factory is _OptionApp:
            raise OptionDataError("ibapi is not installed", connection=True)
        last = "no connection attempt"
        for offset in range(CLIENT_ID_RETRY_COUNT):
            client_id = self.base_client_id + offset
            app = self._app_factory()
            try:
                app.connect(self.host, self.port, client_id)
            except Exception as exc:
                raise OptionDataError(f"IB not connected ({exc})", connection=True) from exc
            if not app.isConnected():
                raise OptionDataError("IB not connected", connection=True)
            thread = threading.Thread(target=app.run, name="options-chase-ib", daemon=True)
            thread.start()
            if app.ready.wait(self._connect_timeout) and not app.client_id_in_use:
                app.reqMarketDataType(MARKET_DATA_TYPE_LIVE)
                self._app, self._thread, self.client_id = app, thread, client_id
                return
            last = "; ".join(app.connection_errors[-2:]) or "no nextValidId"
            self._drop(app, thread)
            if not app.client_id_in_use:
                break
        raise OptionDataError(f"IB not connected (client {self.base_client_id}: {last})",
                              connection=True)

    @staticmethod
    def _drop(app, thread) -> None:
        try:
            app.disconnect()
        except Exception as exc:
            note_swallowed("options chase IB disconnect failed", exc, quiet=True)
        if thread is not None:
            thread.join(timeout=2.0)

    def close(self) -> None:
        with self._lock:
            if self._app is not None:
                self._drop(self._app, self._thread)
            self._app = self._thread = None

    def _req_id(self) -> int:
        req_id = self._next_req_id
        self._next_req_id += 1
        return req_id

    def _pace(self) -> None:
        if self._gap > 0:
            time.sleep(self._gap)

    # --- requests ---
    def _chain_params(self, app, symbol: str, today: date) -> dict[str, Any]:
        """Option parameters for the symbol, fetched once per day."""
        cached = self._params.get(symbol)
        if cached is not None and cached[0] == today:
            return cached[1]
        req_id = self._req_id()
        done = app.expect(req_id)
        app.reqContractDetails(req_id, _stock_contract(symbol))
        done.wait(self._chain_timeout)
        con_ids, _q, errors = app.take(req_id)
        self._pace()
        if not con_ids:
            raise OptionDataError(f"no IB contract ({errors[0]})" if errors else "no IB contract")
        req_id = self._req_id()
        done = app.expect(req_id)
        app.reqSecDefOptParams(req_id, symbol.replace("-", " "), "", "STK", int(con_ids[0]))
        done.wait(self._chain_timeout)
        rows, _q, errors = app.take(req_id)
        self._pace()
        params = select_params(rows, symbol)
        if not params:
            raise OptionDataError("empty chain")
        self._params[symbol] = (today, params)
        return params

    def fetch_chain(self, symbol: str, *, side: str, last: float | None, hv: float | None,
                    today: date) -> dict[str, Any]:
        """Chain snapshot mapping for `options_chase.pick_candidate`. Raises OptionDataError."""
        symbol = str(symbol or "").strip().upper()
        with self._lock:
            if not self._alive():
                if self._app is not None:
                    self._drop(self._app, self._thread)
                    self._app = self._thread = None
                self._connect()
            app = self._app
            assert app is not None
            params = self._chain_params(app, symbol, today)
            expiries = list(params.get("expirations") or ())
            expiry, sessions, _why = options_chase.pick_expiry(expiries, today)
            chain = {"expiries": expiries, "strikes": list(params.get("strikes") or ()), "quotes": []}
            if expiry is None:
                return chain
            right = options_chase.right_for(side)
            plan = options_chase.quote_plan(last, side, chain["strikes"], sessions=sessions, hv=hv)
            requests: list[tuple[int, float, threading.Event]] = []
            for strike in plan:
                req_id = self._req_id()
                done = app.expect(req_id)
                contract = _option_contract(symbol, expiry.isoformat(), strike, right, params)
                # Snapshot, non-regulatory (no per-quote fee), no generic ticks.
                app.reqMktData(req_id, contract, "", True, False, [])
                requests.append((req_id, strike, done))
                self._pace()
            deadline = time.monotonic() + self._quote_timeout
            for _req, _strike, done in requests:
                done.wait(max(0.0, deadline - time.monotonic()))
            denied: set[int] = set()
            for req_id, strike, _done in requests:
                try:
                    app.cancelMktData(req_id)
                except Exception as exc:
                    note_swallowed("options chase snapshot cancel failed", exc, quiet=True)
                _rows, quote, errors = app.take(req_id)
                denied.update(code for code in errors if code in NO_PERMISSION_CODES)
                if quote:
                    chain["quotes"].append({"expiry": expiry.isoformat(), "strike": strike,
                                            "right": right, **quote})
            if denied and not any(q.get("bid") is not None or q.get("ask") is not None
                                  for q in chain["quotes"]):
                raise OptionDataError(
                    f"no option market-data permission (IB {min(denied)})", permission=True)
            return chain


# ------------------------------------------------------------------ the service
def _default_hv(symbol: str, today: date) -> float | None:
    """20-session realized vol from the desk's cached daily bars (read at call time)."""
    import project_paths

    path = Path(project_paths.DAILY_BARS_CACHE_DIR) / f"{symbol}.csv"
    return options_chase.realized_vol(options_chase.read_daily_closes(path, before=today))


def pop_rows(board: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Pop rows of both sides, board rank order (biggest |pop_score| first), each with its side."""
    pop = board.get("pop") or {}
    rows = [dict(row, side=side) for side in ("long", "short") for row in (pop.get(side) or ())]
    return sorted(rows, key=lambda r: -abs(float(r.get("pop_score") or 0.0)))


def row_key(row: Mapping[str, Any]) -> str:
    return f"{str(row.get('symbol') or '').strip().upper()}|{row.get('side') or row.get('_side') or 'long'}"


class OptionsChaseService:
    """Picks, caches and logs chase contracts for the Movers Pop names (worker thread only)."""

    def __init__(
        self,
        *,
        client_factory: Callable[[], Any] | None = None,
        hv_provider: Callable[[str, date], float | None] | None = None,
        log_path: Path | None = None,
        max_per_tick: int = OPTIONS_CHASE_MAX_PER_TICK,
        cache_seconds: float = OPTIONS_CHASE_CACHE_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client_factory = client_factory or IBOptionChainClient
        self._client = None
        self._hv_provider = hv_provider or _default_hv
        if log_path is None:
            from project_paths import OPTIONS_CHASE_LOG_FILE

            log_path = Path(OPTIONS_CHASE_LOG_FILE)
        self._log_path = Path(log_path)
        self._max = int(max_per_tick)
        self._cache_seconds = float(cache_seconds)
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._chains: dict[str, tuple[float, dict[str, Any]]] = {}
        self._hv: dict[str, tuple[date, float | None]] = {}
        self._results: dict[str, dict[str, Any]] = {}
        self._session: date | None = None
        self._down_reason = ""
        self._denied = False
        self._said: set[str] = set()
        self._tracker = options_chase.ChaseOutcomeTracker()
        self.fetches = 0

    # --- reads (any thread) ---
    @property
    def down_reason(self) -> str:
        return self._down_reason

    def status_text(self) -> str:
        return f"options: no option data ({self._down_reason})" if self._down_reason else ""

    def annotate(self, board: Mapping[str, Any]) -> dict[str, Any]:
        """A copy of the board whose Pop rows carry `opt` (result dict or None). Rows are new
        dicts, so a board already handed to the Qt thread is never mutated."""
        out = dict(board)
        pop = board.get("pop") or {}
        if not pop:
            return out
        with self._lock:
            results = dict(self._results)
            down = self._down_reason
        new_pop = {}
        for side in ("long", "short"):
            rows = []
            for row in pop.get(side) or ():
                key = row_key(dict(row, side=side))
                result = results.get(key)
                if result is None and down:
                    result = options_chase.no_data(dict(row, side=side), down)
                rows.append(dict(row, opt=result))
            new_pop[side] = rows
        out["pop"] = {**pop, **new_pop}
        out["options_chase_status"] = down
        return out

    def fresh_mid(self, flag: Mapping[str, Any]) -> float | None:
        """The flagged contract's mid from a cached quote younger than the cache window."""
        key = row_key(flag)
        with self._lock:
            cached = self._chains.get(key)
        if cached is None or self._monotonic() - cached[0] > self._cache_seconds:
            return None
        for quote in cached[1].get("quotes") or ():
            if (quote.get("expiry") == flag.get("expiry") and quote.get("right") == flag.get("right")
                    and options_chase._num(quote.get("strike")) == options_chase._num(flag.get("strike"))):
                bid, ask = options_chase._num(quote.get("bid")), options_chase._num(quote.get("ask"))
                if bid is not None and ask is not None and ask >= bid:
                    return (bid + ask) / 2.0
        return None

    # --- the tick (Movers worker thread) ---
    def _say_once(self, reason: str) -> None:
        kind = reason.split("(")[0].strip()
        if kind not in self._said:
            self._said.add(kind)
            logging.warning("Options chase: no option data this session: %s", reason)

    def _hv_for(self, symbol: str, today: date) -> float | None:
        cached = self._hv.get(symbol)
        if cached is not None and cached[0] == today:
            return cached[1]
        try:
            value = self._hv_provider(symbol, today)
        except Exception:
            value = None
        self._hv[symbol] = (today, value)
        return value

    def _client_or_none(self):
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def run(self, board: Mapping[str, Any], prices: Mapping[str, float], *,
            now: datetime) -> dict[str, dict[str, Any]]:
        """Fetch (capped, cached), pick, log. Returns this tick's results by `SYM|side`."""
        moment = now if now.tzinfo is not None else now.astimezone()
        today = moment.astimezone(options_chase.NY_TZ).date()
        if self._session != today:
            self._session, self._denied, self._said = today, False, set()
            self._down_reason = ""
            self._hv = {}
        eligible = [r for r in pop_rows(board)
                    if (options_chase._num(r.get("rvol")) or 0.0) >= options_chase.MIN_RVOL]
        chosen = eligible[: self._max]
        results: dict[str, dict[str, Any]] = {}
        for row in chosen:
            symbol = str(row.get("symbol") or "").strip().upper()
            pop = dict(row, move15=row.get("move15_pct"))
            hv = self._hv_for(symbol, today)
            key = row_key(row)
            if self._denied:
                results[key] = options_chase.no_data(pop, self._down_reason)
                continue
            chain = self._cached_chain(key)
            if chain is None:
                try:
                    chain = self._client_or_none().fetch_chain(
                        symbol, side=row["side"], last=options_chase._num(row.get("last")),
                        hv=hv, today=today)
                    self.fetches += 1
                    chain["as_of"] = moment.isoformat(timespec="seconds")
                    with self._lock:
                        self._chains[key] = (self._monotonic(), chain)
                    self._down_reason = ""
                except OptionDataError as exc:
                    if exc.permission or exc.connection:
                        self._down_reason = exc.reason
                        self._denied = exc.permission
                        self._say_once(exc.reason)
                        results[key] = options_chase.no_data(pop, exc.reason)
                        if exc.connection:
                            break  # one failed connect per tick, not one per name
                        continue
                    if exc.reason == "empty chain":
                        results[key] = options_chase.pick_candidate(pop, {}, today=today, hv=hv)
                    else:
                        results[key] = options_chase.no_data(pop, exc.reason)
                    continue
                except Exception as exc:  # a broken client never breaks the board
                    logging.warning("Options chase fetch failed for %s: %s", symbol, exc)
                    results[key] = options_chase.no_data(pop, "IB request failed")
                    continue
            result = options_chase.pick_candidate(pop, chain, today=today, hv=hv)
            result["as_of"] = chain.get("as_of")
            results[key] = result
        with self._lock:
            self._results = results
        records = self._tracker.flag(results.values(), now=moment)
        records += self._tracker.observe(prices, now=moment, option_mid=self.fresh_mid)
        options_chase.append_records(self._log_path, records)
        return results

    def _cached_chain(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            cached = self._chains.get(key)
        if cached is None or self._monotonic() - cached[0] >= self._cache_seconds:
            return None
        return cached[1]

    def close(self) -> None:
        """Disconnect (joins a thread: never call on the Qt thread)."""
        client = self._client
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                note_swallowed("options chase client close failed", exc, quiet=True)
