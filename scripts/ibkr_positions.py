"""IBKR book, read-only: accounts, open positions and cash for the Trade Mentor ``/book``.

A short-lived TWS API client on its own fixed client id (:data:`IBKR_BOOK_CLIENT_ID`,
local setting ``mentor_ibkr_client_id`` overrides). One read: ``reqManagedAccts`` ->
``reqPositions`` until ``positionEnd`` -> ``reqAccountSummary`` until
``accountSummaryEnd`` -> cancel both -> ``disconnect``. Nothing else is ever sent: no
order, open-order or market-data request (:data:`ALLOWED_REQUESTS`). Qt-free, no store
writes: the caller keeps the snapshot, the 15-min freshness and the 1-hour backoff.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

from questrade_positions import BookSnapshot

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper

    IBAPI_AVAILABLE = True
    _BASES: tuple[type, ...] = (EWrapper, EClient)
except Exception:  # pragma: no cover - exercised when ibapi is not installed
    EClient = None  # type: ignore[assignment,misc]
    IBAPI_AVAILABLE = False
    _BASES = (object,)

BROKER = "IBKR"
#: The book reader's own fixed TWS client id, clear of 9125-9127 journal, 9135-9137 scanner,
#: 9140-9142 momentum universe and 9145-9147 options chase (their ids plus retries).
IBKR_BOOK_CLIENT_ID = 9155
CLIENT_ID_SETTING = "mentor_ibkr_client_id"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7496
TIMEOUT_SEC = 15.0
SUMMARY_REQ_ID = 9155
#: Totals in the base currency, plus ``$LEDGER:ALL`` for cash in each currency held.
SUMMARY_TAGS = "TotalCashValue,NetLiquidation,AvailableFunds,BuyingPower,$LEDGER:ALL"
#: The only TWS requests this module may send (plus connect / run / isConnected / disconnect).
ALLOWED_REQUESTS = frozenset({
    "reqManagedAccts", "reqPositions", "cancelPositions", "reqAccountSummary", "cancelAccountSummary",
})
#: Farm / connectivity notices, not errors.
_INFO_CODES = frozenset({2103, 2104, 2105, 2106, 2107, 2108, 2119, 2158})
_OPTION_TYPES = frozenset({"OPT", "FOP", "WAR"})


def _aware(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.astimezone()


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN is unknown


class IBKRPositionsReader(*_BASES):  # type: ignore[misc]
    """One read of positions and account summary; build a fresh reader per read."""

    def __init__(self) -> None:
        if not IBAPI_AVAILABLE:
            raise RuntimeError("ibapi is not installed")
        EClient.__init__(self, self)
        self.managed_ready = threading.Event()
        self.positions_done = threading.Event()
        self.summary_done = threading.Event()
        self.managed: list[str] = []
        self.raw_positions: list[dict[str, Any]] = []
        self.summary: list[tuple[str, str, str, str]] = []
        self.errors: list[str] = []
        self.closed = False

    # ------------------------------------------------------------ callbacks (reader thread)
    def managedAccounts(self, accountsList: str) -> None:  # noqa: N802,N803 - ibapi names
        self.managed = [a.strip() for a in str(accountsList or "").split(",") if a.strip()]
        self.managed_ready.set()

    def position(self, account: str, contract: Any, position: Any, avgCost: float) -> None:  # noqa: N803
        self.raw_positions.append({"account": str(account or ""), "contract": contract,
                                   "position": position, "avg_cost": avgCost})

    def positionEnd(self) -> None:  # noqa: N802
        self.positions_done.set()

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str) -> None:  # noqa: N802,N803
        self.summary.append((str(account or ""), str(tag or ""), str(value or ""), str(currency or "")))

    def accountSummaryEnd(self, reqId: int) -> None:  # noqa: N802,N803
        self.summary_done.set()

    def error(self, reqId: int, errorCode: int, errorString: str, *args: Any) -> None:  # noqa: N802,N803
        if errorCode not in _INFO_CODES:
            self.errors.append(f"{errorCode}: {errorString}"[:160])

    def connectionClosed(self) -> None:  # noqa: N802
        self.closed = True
        self.managed_ready.set()
        self.positions_done.set()
        self.summary_done.set()

    # ------------------------------------------------------------ the one read
    def fetch(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, client_id: int = IBKR_BOOK_CLIENT_ID,
              timeout_sec: float = TIMEOUT_SEC, *, now: datetime | None = None,
              journal_accounts: Iterable[Mapping[str, Any]] = ()) -> tuple[BookSnapshot | None, str]:
        """``(snapshot, "")`` or ``(None, reason)``. Never raises; always disconnects."""
        moment = _aware(now)
        wait_for = max(0.05, float(timeout_sec))
        deadline = time.monotonic() + wait_for

        def left() -> float:
            return max(0.0, deadline - time.monotonic())

        thread: threading.Thread | None = None
        asked: set[str] = set()
        summary_ok = False
        try:
            # ibapi's connect can loop forever when TWS accepts the socket and never answers (mid-login,
            # frozen), so it runs on its own daemon thread inside the same deadline.
            dialled = threading.Event()
            failure: list[BaseException] = []

            def dial() -> None:
                try:
                    self.connect(host, int(port), int(client_id))
                except Exception as exc:  # noqa: BLE001 - reported below
                    failure.append(exc)
                finally:
                    dialled.set()

            threading.Thread(target=dial, daemon=True, name="ibkr-book-connect").start()
            if not dialled.wait(left()):
                return None, f"TWS not answering (connect did not finish in {wait_for:g}s)"
            if failure:  # a refused socket is "TWS not running"
                return None, f"TWS not running ({type(failure[0]).__name__})"
            if not self.isConnected():
                return None, "TWS not running" + (f" ({self.errors[-1]})" if self.errors else "")
            thread = threading.Thread(target=self.run, daemon=True, name="ibkr-book-reader")
            thread.start()
            self.reqManagedAccts()
            self.managed_ready.wait(min(2.0, left()))
            asked.add("positions")
            self.reqPositions()
            done = self.positions_done.wait(left())
            detail = f"; last error {self.errors[-1]}" if self.errors else ""
            if self.closed:
                return None, f"TWS closed the connection{detail}"
            if not done:
                return None, f"timeout: no positionEnd after {wait_for:g}s{detail}"
            asked.add("summary")
            self.reqAccountSummary(SUMMARY_REQ_ID, "All", SUMMARY_TAGS)
            summary_ok = self.summary_done.wait(left()) and not self.closed
        except Exception as exc:  # noqa: BLE001 - one failure = no snapshot and a backoff, never a raise
            return None, f"{type(exc).__name__}: {exc}"[:200]
        finally:
            self._close(thread, asked)
        errors = [] if summary_ok else ["account summary: no accountSummaryEnd (cash unknown)"]
        snap = build_snapshot(moment, self.managed, self.raw_positions, self.summary if summary_ok else [],
                              journal_accounts, errors)
        return snap, ""

    def _close(self, thread: threading.Thread | None, asked: set[str]) -> None:
        try:
            if self.isConnected():
                if "positions" in asked:
                    self.cancelPositions()
                if "summary" in asked:
                    self.cancelAccountSummary(SUMMARY_REQ_ID)
        except Exception as exc:  # noqa: BLE001 - the disconnect below still runs
            logging.debug("Trade Mentor book: IBKR cancel failed (%s)", exc)
        try:
            self.disconnect()
        except Exception as exc:  # noqa: BLE001
            logging.debug("Trade Mentor book: IBKR disconnect failed (%s)", exc)
        if thread is not None:
            thread.join(timeout=2.0)


# ---------------------------------------------------------------- mapping to the book shape
def _journal_by_number(journal_accounts: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in journal_accounts or ():
        broker = str(row.get("broker") or "").upper()
        number = str(row.get("account_number") or "").strip()
        if number and (broker.startswith(BROKER) or number not in out):
            out[number] = row
    return out


def position_row(raw: Mapping[str, Any], account: Mapping[str, Any]) -> dict[str, Any] | None:
    """One ``position`` callback in the book shape; a flat (0 qty) row is None."""
    contract = raw.get("contract")
    qty = _float(raw.get("position"))
    if not qty:
        return None
    sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
    option = sec_type in _OPTION_TYPES
    symbol = str((getattr(contract, "localSymbol", "") if option else "") or getattr(contract, "symbol", "") or "")
    multiplier = _float(getattr(contract, "multiplier", None) or None)
    avg_cost = _float(raw.get("avg_cost"))
    # IB's avgCost for a contract includes the multiplier; the book shows the per-unit price.
    avg = avg_cost / multiplier if avg_cost is not None and multiplier and multiplier != 1 else avg_cost
    return {
        "broker": BROKER,
        "account_number": str(raw.get("account") or ""),
        "account_label": str(account.get("account_label") or raw.get("account") or ""),
        "account_type": str(account.get("account_type") or ""),
        "symbol": " ".join(symbol.split()).upper(),
        "security_type": sec_type,
        "currency": str(getattr(contract, "currency", "") or "").upper(),
        "multiplier": multiplier if multiplier is not None else (100.0 if option else None),
        "open_qty": abs(qty),
        "side": "LONG" if qty > 0 else "SHORT",
        "avg_price": avg,
        "current_price": None,  # no market data is ever requested
        "market_value": None,
    }


def _by_currency(summary: Iterable[tuple[str, str, str, str]], account: str, tag: str) -> dict[str, float] | None:
    out: dict[str, float] = {}
    for acct, name, value, currency in summary:
        cur = currency.strip().upper()
        number = _float(value)
        if acct == account and name == tag and cur and cur != "BASE" and number is not None:
            out[cur] = number
    return out or None


def account_row(number: str, own: Mapping[str, Any], summary: list[tuple[str, str, str, str]]) -> dict[str, Any]:
    cash = _by_currency(summary, number, "CashBalance") or _by_currency(summary, number, "TotalCashValue")
    return {
        "broker": BROKER, "account_number": number,
        "account_type": str(own.get("account_type") or ""),
        "account_label": str(own.get("account_label") or "").strip() or number,
        "status": "", "cash": cash, "cash_known": cash is not None,
        "net_liquidation": _by_currency(summary, number, "NetLiquidation"),
        "available_funds": _by_currency(summary, number, "AvailableFunds"),
        "buying_power": _by_currency(summary, number, "BuyingPower"),
    }


def build_snapshot(moment: datetime, managed: Iterable[str], raw_positions: Iterable[Mapping[str, Any]],
                   summary: list[tuple[str, str, str, str]], journal_accounts: Iterable[Mapping[str, Any]] = (),
                   errors: Iterable[str] = ()) -> BookSnapshot:
    """The IBKR read as a ``BookSnapshot`` (``broker="IBKR"``); labels from the journal accounts when known."""
    own = _journal_by_number(journal_accounts)
    raws = list(raw_positions)
    numbers: list[str] = []
    for number in [*managed, *(str(r.get("account") or "") for r in raws), *(row[0] for row in summary)]:
        if number and number not in numbers:
            numbers.append(number)
    accounts = [account_row(number, own.get(number) or {}, summary) for number in numbers]
    by_number = {a["account_number"]: a for a in accounts}
    positions = [row for raw in raws if (row := position_row(raw, by_number.get(str(raw.get("account")), {})))]
    stamp = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    return BookSnapshot(stamp, tuple(accounts), tuple(positions), tuple(errors), broker=BROKER)


# ---------------------------------------------------------------- the caller's entry point
def _settings() -> tuple[str, int, int]:
    from project_paths import get_local_setting

    host = str(get_local_setting("journal_ibkr_host", DEFAULT_HOST) or DEFAULT_HOST)
    port = int(_float(get_local_setting("journal_ibkr_port", DEFAULT_PORT)) or DEFAULT_PORT)
    client_id = int(_float(get_local_setting(CLIENT_ID_SETTING, IBKR_BOOK_CLIENT_ID)) or IBKR_BOOK_CLIENT_ID)
    return host, port, client_id


def _live_journal_accounts() -> list[dict[str, Any]]:
    from pathlib import Path

    from project_paths import JOURNAL_DB_FILE

    from mentor_packs.book_pack import read_accounts

    return read_accounts(Path(JOURNAL_DB_FILE))


def fetch_book(now: datetime | None = None, *, reader_factory: Callable[[], Any] | None = None,
               journal_accounts: Callable[[], Iterable[Mapping[str, Any]]] | None = None,
               timeout_sec: float = TIMEOUT_SEC) -> tuple[BookSnapshot | None, str]:
    """``(snapshot, "")`` or ``(None, reason)``. Never raises. Same contract as Questrade's ``fetch_book``."""
    moment = _aware(now)
    try:
        host, port, client_id = _settings()
    except Exception as exc:  # noqa: BLE001 - unreadable settings = the defaults
        logging.debug("Trade Mentor book: IBKR settings unreadable (%s)", exc)
        host, port, client_id = DEFAULT_HOST, DEFAULT_PORT, IBKR_BOOK_CLIENT_ID
    try:
        accounts = list((journal_accounts or _live_journal_accounts)() or ())
    except Exception:  # noqa: BLE001 - labels are optional; the account number stands in
        accounts = []
    try:
        reader = (reader_factory or IBKRPositionsReader)()
    except Exception as exc:  # noqa: BLE001
        return None, f"IBKR reader unavailable ({type(exc).__name__}: {exc})"[:200]
    try:
        snap, reason = reader.fetch(host, port, client_id, timeout_sec, now=moment, journal_accounts=accounts)
    except Exception as exc:  # noqa: BLE001 - the reader never raises; a stub might
        snap, reason = None, f"{type(exc).__name__}: {exc}"[:200]
    if snap is None:
        logging.warning("Trade Mentor book: IBKR read failed (%s)", reason)
        return None, str(reason or "unknown")
    logging.info("Trade Mentor book: IBKR read %d account(s), %d position(s) on client id %d",
                 len(snap.accounts), len(snap.positions), client_id)
    return snap, ""
