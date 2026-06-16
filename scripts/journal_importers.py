from __future__ import annotations

import csv
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any
import zoneinfo

import requests

from project_paths import get_local_setting, save_local_setting


PACIFIC_TZ_NAME = "America/Vancouver"

QUESTRADE_REFRESH_TOKEN_SETTING = "journal_questrade_refresh_token"
QUESTRADE_ACCESS_TOKEN_SETTING = "journal_questrade_access_token"
QUESTRADE_API_SERVER_SETTING = "journal_questrade_api_server"
QUESTRADE_EXPIRES_AT_SETTING = "journal_questrade_expires_at"
QUESTRADE_LOGIN_URL = "https://login.questrade.com/oauth2/token"

IBKR_HOST_SETTING = "journal_ibkr_host"
IBKR_PORT_SETTING = "journal_ibkr_port"
IBKR_CLIENT_ID_SETTING = "journal_ibkr_client_id"
IBKR_ENABLED_SETTING = "journal_ibkr_enabled"
IBKR_DEFAULT_CLIENT_ID = 9125
IBKR_CLIENT_ID_RETRY_COUNT = 3


try:
    from ibapi.client import EClient
    from ibapi.contract import Contract
    from ibapi.execution import ExecutionFilter
    from ibapi.wrapper import EWrapper

    IBAPI_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when ibapi is not installed.
    EClient = object  # type: ignore
    EWrapper = object  # type: ignore
    Contract = object  # type: ignore
    ExecutionFilter = object  # type: ignore
    IBAPI_AVAILABLE = False


def pacific_now() -> datetime:
    return datetime.now(zoneinfo.ZoneInfo(PACIFIC_TZ_NAME))


def resolve_ibkr_client_id(value: Any | None = None) -> int:
    if value is None:
        value = get_local_setting(IBKR_CLIENT_ID_SETTING, IBKR_DEFAULT_CLIENT_ID)
    try:
        client_id = int(value or IBKR_DEFAULT_CLIENT_ID)
    except (TypeError, ValueError):
        client_id = IBKR_DEFAULT_CLIENT_ID
    return client_id if client_id > 0 else IBKR_DEFAULT_CLIENT_ID


def _is_ibkr_client_id_conflict(message: Any) -> bool:
    text = str(message or "").lower()
    return "326" in text or "client id" in text and "already in use" in text


def pacific_day_bounds(target_date: date | None = None) -> tuple[datetime, datetime]:
    local_tz = zoneinfo.ZoneInfo(PACIFIC_TZ_NAME)
    day = target_date or pacific_now().date()
    start = datetime.combine(day, dt_time.min, tzinfo=local_tz)
    end = datetime.combine(day, dt_time.max.replace(microsecond=0), tzinfo=local_tz)
    return start, end


def chunk_date_ranges(start_date: date, end_date: date, max_days: int = 31) -> list[tuple[date, date]]:
    if end_date < start_date:
        return []
    chunks = []
    cursor = start_date
    step = max(1, int(max_days))
    while cursor <= end_date:
        chunk_end = min(end_date, cursor + timedelta(days=step - 1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def mask_secret(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    if len(cleaned) <= 8:
        return "*" * len(cleaned)
    return f"{cleaned[:4]}...{cleaned[-4:]}"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_date(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def parse_broker_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return pacific_now()
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    for fmt in ("%Y%m%d  %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=zoneinfo.ZoneInfo(PACIFIC_TZ_NAME))
        except ValueError:
            continue
    return pacific_now()


def normalize_side(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"BUY", "BOT", "BTO", "COVER", "BUYTOCOVER"}:
        return "BUY"
    if text in {"SELL", "SLD", "STC", "SSHORT", "SELLSHORT"}:
        return "SELL"
    if text in {"LONG"}:
        return "BUY"
    if text in {"SHORT"}:
        return "SELL"
    return text or "BUY"


@dataclass
class NormalizedExecution:
    execution_uid: str
    broker: str
    account_number: str
    account_label: str
    account_type: str
    symbol: str
    security_type: str
    currency: str
    side: str
    quantity: float
    price: float
    timestamp: str
    trade_date: str
    commission: float = 0.0
    fees: float = 0.0
    gross_amount: float | None = None
    net_amount: float | None = None
    order_id: str = ""
    exchange_exec_id: str = ""
    raw_json: str = "{}"

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def _execution_uid(prefix: str, *parts: Any) -> str:
    cleaned = [str(part or "").strip() for part in parts if str(part or "").strip()]
    if cleaned:
        return f"{prefix}:{':'.join(cleaned)}"
    return f"{prefix}:{uuid.uuid4().hex}"


class QuestradeImporter:
    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    @property
    def refresh_token(self) -> str:
        return (
            os.environ.get("QUESTRADE_REFRESH_TOKEN")
            or str(get_local_setting(QUESTRADE_REFRESH_TOKEN_SETTING, "") or "")
        ).strip()

    @property
    def access_token(self) -> str:
        return (
            os.environ.get("QUESTRADE_ACCESS_TOKEN")
            or str(get_local_setting(QUESTRADE_ACCESS_TOKEN_SETTING, "") or "")
        ).strip()

    @property
    def api_server(self) -> str:
        value = (
            os.environ.get("QUESTRADE_API_SERVER")
            or str(get_local_setting(QUESTRADE_API_SERVER_SETTING, "") or "")
        ).strip()
        if not value:
            return ""
        return value if value.endswith("/") else value + "/"

    def status_lines(self) -> list[str]:
        lines = []
        refresh_token = self.refresh_token
        access_token = self.access_token
        api_server = self.api_server
        lines.append(f"Refresh token: {mask_secret(refresh_token) if refresh_token else 'not set'}")
        lines.append(f"Access token: {mask_secret(access_token) if access_token else 'not set'}")
        lines.append(f"API server: {api_server or 'not set'}")
        if os.environ.get("QUESTRADE_REFRESH_TOKEN") or os.environ.get("QUESTRADE_ACCESS_TOKEN"):
            lines.append("Environment Questrade token values are taking priority.")
        expires_at = str(get_local_setting(QUESTRADE_EXPIRES_AT_SETTING, "") or "").strip()
        if expires_at:
            lines.append(f"Saved token expiry: {expires_at}")
        return lines

    def refresh_access_token(self) -> dict[str, Any]:
        refresh_token = self.refresh_token
        if not refresh_token:
            raise RuntimeError("Questrade refresh token is not configured.")
        response = self.session.get(
            QUESTRADE_LOGIN_URL,
            params={"grant_type": "refresh_token", "refresh_token": refresh_token},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        access_token = str(payload.get("access_token") or "").strip()
        api_server = str(payload.get("api_server") or "").strip()
        next_refresh = str(payload.get("refresh_token") or "").strip()
        expires_in = int(payload.get("expires_in", 0) or 0)
        expires_at = (datetime.now() + timedelta(seconds=max(0, expires_in))).isoformat(timespec="seconds")

        if access_token and not os.environ.get("QUESTRADE_ACCESS_TOKEN"):
            save_local_setting(QUESTRADE_ACCESS_TOKEN_SETTING, access_token)
        if api_server and not os.environ.get("QUESTRADE_API_SERVER"):
            save_local_setting(QUESTRADE_API_SERVER_SETTING, api_server)
        if next_refresh and not os.environ.get("QUESTRADE_REFRESH_TOKEN"):
            save_local_setting(QUESTRADE_REFRESH_TOKEN_SETTING, next_refresh)
        save_local_setting(QUESTRADE_EXPIRES_AT_SETTING, expires_at)
        return payload

    def _authorized_get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.access_token or not self.api_server:
            self.refresh_access_token()
        url = self.api_server + path.lstrip("/")
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = self.session.get(url, headers=headers, params=params or {}, timeout=30)
        if response.status_code == 401 and self.refresh_token:
            self.refresh_access_token()
            url = self.api_server + path.lstrip("/")
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.get(url, headers=headers, params=params or {}, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_accounts(self) -> list[dict[str, Any]]:
        payload = self._authorized_get("v1/accounts")
        accounts = payload.get("accounts", []) if isinstance(payload, dict) else []
        return [dict(item) for item in accounts if isinstance(item, dict)]

    def get_executions(self, account_number: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        payload = self._authorized_get(
            f"v1/accounts/{account_number}/executions",
            params={"startTime": start.isoformat(), "endTime": end.isoformat()},
        )
        executions = payload.get("executions", []) if isinstance(payload, dict) else []
        return [dict(item) for item in executions if isinstance(item, dict)]

    def get_activities(self, account_number: str, start_date: date, end_date: date) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for chunk_start, chunk_end in chunk_date_ranges(start_date, end_date, max_days=31):
            payload = self._authorized_get(
                f"v1/accounts/{account_number}/activities",
                params={"startTime": chunk_start.isoformat(), "endTime": chunk_end.isoformat()},
            )
            activities = payload.get("activities", []) if isinstance(payload, dict) else []
            rows.extend(dict(item) for item in activities if isinstance(item, dict))
        return rows

    def normalize_execution(self, raw: dict[str, Any], account: dict[str, Any]) -> NormalizedExecution:
        timestamp = parse_broker_datetime(
            raw.get("timestamp") or raw.get("executionTime") or raw.get("time") or raw.get("tradeDate")
        )
        account_number = str(account.get("number") or account.get("accountNumber") or raw.get("accountNumber") or "").strip()
        account_type = str(account.get("type") or account.get("accountType") or "").strip()
        account_label = str(account.get("name") or account.get("description") or account_type or account_number).strip()
        symbol = str(raw.get("symbol") or raw.get("symbolName") or raw.get("underlyingSymbol") or "").strip().upper()
        security_type = str(raw.get("securityType") or raw.get("symbolType") or raw.get("listingExchange") or "UNKNOWN").strip().upper()
        currency = str(raw.get("currency") or account.get("currency") or "USD").strip().upper()
        side = normalize_side(raw.get("side") or raw.get("action"))
        quantity = abs(_coerce_float(raw.get("quantity") or raw.get("shares")))
        price = _coerce_float(raw.get("price") or raw.get("executionPrice"))
        commission = abs(_coerce_float(raw.get("commission")))
        fees = abs(
            _coerce_float(raw.get("fees"))
            + _coerce_float(raw.get("secFee"))
            + _coerce_float(raw.get("exchangeFee"))
            + _coerce_float(raw.get("ecnFee"))
        )
        execution_id = raw.get("id") or raw.get("executionId") or raw.get("execId")
        order_id = str(raw.get("orderId") or raw.get("orderID") or "").strip()
        exchange_exec_id = str(raw.get("exchangeExecId") or raw.get("venueExecId") or execution_id or "").strip()
        gross_amount = raw.get("grossAmount") if "grossAmount" in raw else None
        net_amount = raw.get("netAmount") if "netAmount" in raw else None
        return NormalizedExecution(
            execution_uid=_execution_uid("QT", account_number, execution_id or order_id, symbol, timestamp.isoformat()),
            broker="QUESTRADE",
            account_number=account_number,
            account_label=account_label,
            account_type=account_type,
            symbol=symbol,
            security_type=security_type,
            currency=currency,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp.isoformat(),
            trade_date=timestamp.date().isoformat(),
            commission=commission,
            fees=fees,
            gross_amount=_coerce_float(gross_amount, default=0.0) if gross_amount is not None else None,
            net_amount=_coerce_float(net_amount, default=0.0) if net_amount is not None else None,
            order_id=order_id,
            exchange_exec_id=exchange_exec_id,
            raw_json=json.dumps(raw, sort_keys=True, default=str),
        )

    def import_executions_for_date(self, target_date: date | None = None) -> tuple[list[NormalizedExecution], list[dict[str, Any]]]:
        start, end = pacific_day_bounds(target_date)
        accounts = self.get_accounts()
        executions: list[NormalizedExecution] = []
        for account in accounts:
            account_number = str(account.get("number") or account.get("accountNumber") or "").strip()
            if not account_number:
                continue
            for raw in self.get_executions(account_number, start, end):
                executions.append(self.normalize_execution(raw, account))
        return executions, accounts


class IBKRExecutionImporter(EWrapper, EClient):  # type: ignore[misc]
    def __init__(self) -> None:
        if not IBAPI_AVAILABLE:
            raise RuntimeError("ibapi is not installed.")
        EClient.__init__(self, self)
        self.exec_end = threading.Event()
        self.executions: list[dict[str, Any]] = []
        self.commissions: dict[str, dict[str, Any]] = {}
        self.errors: list[str] = []

    def execDetails(self, reqId: int, contract: Contract, execution: Any) -> None:  # noqa: N802 - ibapi callback name
        self.executions.append({"contract": contract, "execution": execution})

    def execDetailsEnd(self, reqId: int) -> None:  # noqa: N802 - ibapi callback name
        self.exec_end.set()

    def commissionReport(self, commissionReport: Any) -> None:  # noqa: N802 - ibapi callback name
        exec_id = str(getattr(commissionReport, "execId", "") or "")
        if exec_id:
            self.commissions[exec_id] = {
                "commission": _coerce_float(getattr(commissionReport, "commission", 0.0)),
                "currency": str(getattr(commissionReport, "currency", "") or "").upper(),
                "realized_pnl": _coerce_float(getattr(commissionReport, "realizedPNL", 0.0)),
                "yield": _coerce_float(getattr(commissionReport, "yield_", 0.0)),
            }

    def commissionAndFeesReport(self, report: Any) -> None:  # noqa: N802 - newer callback name
        self.commissionReport(report)

    def error(self, reqId: int, errorCode: int, errorString: str, *args: Any) -> None:  # noqa: N802
        if errorCode not in (2104, 2106, 2158):
            self.errors.append(f"{errorCode}[{reqId}]: {errorString}")

    def fetch(self, *, host: str, port: int, client_id: int, account: str = "", timeout_sec: float = 20.0) -> list[NormalizedExecution]:
        self.connect(host, int(port), int(client_id))
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        time.sleep(0.5)

        filter_obj = ExecutionFilter()
        if account:
            filter_obj.acctCode = account
        self.exec_end.clear()
        self.reqExecutions(7001, filter_obj)
        self.exec_end.wait(timeout=max(1.0, float(timeout_sec)))
        try:
            self.disconnect()
        finally:
            thread.join(timeout=2.0)
        if self.errors and not self.executions:
            raise RuntimeError("; ".join(self.errors[-3:]))
        return [self.normalize_execution(item["contract"], item["execution"]) for item in self.executions]

    def normalize_execution(self, contract: Contract, execution: Any) -> NormalizedExecution:
        timestamp = parse_broker_datetime(getattr(execution, "time", ""))
        exec_id = str(getattr(execution, "execId", "") or "")
        account_number = str(getattr(execution, "acctNumber", "") or "")
        security_type = str(getattr(contract, "secType", "") or "UNKNOWN").upper()
        symbol = str(
            (getattr(contract, "localSymbol", "") if security_type in {"OPT", "FOP", "WAR"} else "")
            or getattr(contract, "symbol", "")
            or getattr(execution, "symbol", "")
            or ""
        ).strip().upper()
        currency = str(getattr(contract, "currency", "") or "").upper()
        commission_report = self.commissions.get(exec_id, {})
        if not currency:
            currency = str(commission_report.get("currency") or "USD").upper()
        return NormalizedExecution(
            execution_uid=_execution_uid("IBKR", account_number, exec_id, symbol, timestamp.isoformat()),
            broker="IBKR",
            account_number=account_number,
            account_label=account_number or "IBKR",
            account_type="",
            symbol=symbol,
            security_type=security_type,
            currency=currency or "USD",
            side=normalize_side(getattr(execution, "side", "")),
            quantity=abs(_coerce_float(getattr(execution, "shares", 0.0))),
            price=_coerce_float(getattr(execution, "price", 0.0)),
            timestamp=timestamp.isoformat(),
            trade_date=timestamp.date().isoformat(),
            commission=abs(_coerce_float(commission_report.get("commission"))),
            fees=0.0,
            gross_amount=None,
            net_amount=None,
            order_id=str(getattr(execution, "orderId", "") or ""),
            exchange_exec_id=exec_id,
            raw_json=json.dumps(
                {
                    "contract": {
                        "symbol": symbol,
                        "secType": security_type,
                        "currency": currency,
                        "exchange": str(getattr(contract, "exchange", "") or ""),
                    },
                    "execution": {
                        "execId": exec_id,
                        "time": str(getattr(execution, "time", "") or ""),
                        "acctNumber": account_number,
                        "side": str(getattr(execution, "side", "") or ""),
                        "shares": _coerce_float(getattr(execution, "shares", 0.0)),
                        "price": _coerce_float(getattr(execution, "price", 0.0)),
                        "orderId": str(getattr(execution, "orderId", "") or ""),
                    },
                    "commission": commission_report,
                },
                sort_keys=True,
                default=str,
            ),
        )


def import_ibkr_executions(
    *,
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
    account: str = "",
    timeout_sec: float = 20.0,
) -> list[NormalizedExecution]:
    if not IBAPI_AVAILABLE:
        raise RuntimeError("ibapi is not installed.")
    resolved_host = str(host or get_local_setting(IBKR_HOST_SETTING, "127.0.0.1") or "127.0.0.1")
    resolved_port = int(port or get_local_setting(IBKR_PORT_SETTING, 7496) or 7496)
    resolved_client_id = resolve_ibkr_client_id(client_id)
    last_error: RuntimeError | None = None
    for offset in range(IBKR_CLIENT_ID_RETRY_COUNT):
        candidate_client_id = resolved_client_id + offset
        try:
            return IBKRExecutionImporter().fetch(
                host=resolved_host,
                port=resolved_port,
                client_id=candidate_client_id,
                account=account,
                timeout_sec=timeout_sec,
            )
        except RuntimeError as exc:
            last_error = exc
            if not _is_ibkr_client_id_conflict(exc) or offset >= IBKR_CLIENT_ID_RETRY_COUNT - 1:
                raise
    if last_error is not None:
        raise last_error
    return []


def manual_execution_from_fields(fields: dict[str, Any]) -> NormalizedExecution:
    timestamp = parse_broker_datetime(fields.get("timestamp") or pacific_now())
    broker = str(fields.get("broker") or "MANUAL").strip().upper()
    account_number = str(fields.get("account_number") or "MANUAL").strip()
    symbol = str(fields.get("symbol") or "").strip().upper()
    execution_id = str(fields.get("execution_id") or uuid.uuid4().hex).strip()
    return NormalizedExecution(
        execution_uid=_execution_uid(broker, account_number, execution_id),
        broker=broker,
        account_number=account_number,
        account_label=str(fields.get("account_label") or account_number).strip(),
        account_type=str(fields.get("account_type") or "").strip(),
        symbol=symbol,
        security_type=str(fields.get("security_type") or "STK").strip().upper(),
        currency=str(fields.get("currency") or "USD").strip().upper(),
        side=normalize_side(fields.get("side")),
        quantity=abs(_coerce_float(fields.get("quantity"))),
        price=_coerce_float(fields.get("price")),
        timestamp=timestamp.isoformat(),
        trade_date=timestamp.date().isoformat(),
        commission=abs(_coerce_float(fields.get("commission"))),
        fees=abs(_coerce_float(fields.get("fees"))),
        gross_amount=None,
        net_amount=None,
        order_id=str(fields.get("order_id") or "").strip(),
        exchange_exec_id=execution_id,
        raw_json=json.dumps(fields, sort_keys=True, default=str),
    )


def parse_csv_executions(path: Path, *, default_broker: str = "MANUAL") -> list[NormalizedExecution]:
    executions: list[NormalizedExecution] = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = {str(key or "").strip().lower(): value for key, value in row.items()}
            fields = {
                "broker": normalized.get("broker") or default_broker,
                "account_number": normalized.get("account") or normalized.get("account_number") or normalized.get("accountnumber"),
                "account_label": normalized.get("account_label") or normalized.get("account_name"),
                "account_type": normalized.get("account_type"),
                "symbol": normalized.get("symbol") or normalized.get("ticker"),
                "security_type": normalized.get("security_type") or normalized.get("sectype") or normalized.get("asset_type"),
                "currency": normalized.get("currency"),
                "side": normalized.get("side") or normalized.get("action"),
                "quantity": normalized.get("quantity") or normalized.get("shares") or normalized.get("qty"),
                "price": normalized.get("price") or normalized.get("execution_price"),
                "timestamp": normalized.get("timestamp") or normalized.get("time") or normalized.get("datetime") or normalized.get("date"),
                "commission": normalized.get("commission"),
                "fees": normalized.get("fees"),
                "order_id": normalized.get("order_id") or normalized.get("orderid"),
                "execution_id": normalized.get("execution_id") or normalized.get("exec_id") or normalized.get("id"),
            }
            if fields["symbol"] and fields["quantity"] and fields["price"]:
                executions.append(manual_execution_from_fields(fields))
    return executions
