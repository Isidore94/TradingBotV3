from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from journal_analytics import AutoTagger
from project_paths import JOURNAL_DB_FILE, JOURNAL_EXPORT_DIR


JOURNAL_SCHEMA_VERSION = 1
EPSILON = 0.0000001


REGIME_PRESETS = {
    "mid_term": [
        "Bull trend",
        "Bear trend",
        "Range",
        "Distribution",
        "Accumulation",
        "Volatile transition",
        "Custom",
    ],
    "short_term": [
        "Risk on",
        "Risk off",
        "Chop",
        "Pullback",
        "Breakout",
        "Breakdown",
        "Custom",
    ],
    "intraday": [
        "Trend up",
        "Trend down",
        "Two-way range",
        "Opening drive",
        "Fade day",
        "Low liquidity",
        "Custom",
    ],
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _date_text(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    if len(text) >= 10:
        return text[:10]
    return text


def _row_to_dict(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    return {key: row[key] for key in row.keys()}


def _signed_quantity(row: dict[str, Any]) -> float:
    side = str(row.get("side") or "").strip().upper()
    qty = abs(_coerce_float(row.get("quantity")))
    if side in {"SELL", "SLD", "STC", "SSHORT", "SHORT"}:
        return -qty
    return qty


def _contract_multiplier(row: dict[str, Any]) -> float:
    security_type = str(row.get("security_type") or "").upper()
    raw = {}
    try:
        raw = json.loads(row.get("raw_json") or "{}")
    except Exception:
        raw = {}
    for candidate in (
        raw.get("multiplier") if isinstance(raw, dict) else None,
        (raw.get("contract") or {}).get("multiplier") if isinstance(raw, dict) and isinstance(raw.get("contract"), dict) else None,
    ):
        value = _coerce_float(candidate, default=0.0)
        if value > 0:
            return value
    if security_type in {"OPT", "OPTION", "OPTIONS"}:
        return 100.0
    return 1.0


def _hash_id(*parts: Any) -> str:
    blob = "|".join(str(part or "") for part in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


class JournalStore:
    def __init__(self, db_path: Path = JOURNAL_DB_FILE) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def connection(self):
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS accounts (
                    broker TEXT NOT NULL,
                    account_number TEXT NOT NULL,
                    account_label TEXT NOT NULL DEFAULT '',
                    account_type TEXT NOT NULL DEFAULT '',
                    currency TEXT NOT NULL DEFAULT '',
                    raw_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (broker, account_number)
                );

                CREATE TABLE IF NOT EXISTS import_runs (
                    import_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    imported_executions INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS raw_executions (
                    execution_uid TEXT PRIMARY KEY,
                    broker TEXT NOT NULL,
                    account_number TEXT NOT NULL,
                    account_label TEXT NOT NULL DEFAULT '',
                    account_type TEXT NOT NULL DEFAULT '',
                    symbol TEXT NOT NULL,
                    security_type TEXT NOT NULL DEFAULT '',
                    currency TEXT NOT NULL DEFAULT 'USD',
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    commission REAL NOT NULL DEFAULT 0,
                    fees REAL NOT NULL DEFAULT 0,
                    gross_amount REAL,
                    net_amount REAL,
                    order_id TEXT NOT NULL DEFAULT '',
                    exchange_exec_id TEXT NOT NULL DEFAULT '',
                    raw_json TEXT NOT NULL DEFAULT '{}',
                    imported_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    broker TEXT NOT NULL,
                    account_number TEXT NOT NULL,
                    account_label TEXT NOT NULL DEFAULT '',
                    symbol TEXT NOT NULL,
                    security_type TEXT NOT NULL DEFAULT '',
                    currency TEXT NOT NULL DEFAULT 'USD',
                    direction TEXT NOT NULL,
                    status TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT NOT NULL DEFAULT '',
                    trade_date TEXT NOT NULL,
                    quantity_opened REAL NOT NULL DEFAULT 0,
                    quantity_closed REAL NOT NULL DEFAULT 0,
                    average_entry_price REAL NOT NULL DEFAULT 0,
                    average_exit_price REAL NOT NULL DEFAULT 0,
                    gross_pnl REAL NOT NULL DEFAULT 0,
                    commission REAL NOT NULL DEFAULT 0,
                    fees REAL NOT NULL DEFAULT 0,
                    net_pnl REAL NOT NULL DEFAULT 0,
                    pnl_usd REAL,
                    auto_tag_summary TEXT NOT NULL DEFAULT '',
                    tag_confidence REAL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_legs (
                    leg_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    execution_uid TEXT NOT NULL,
                    side TEXT NOT NULL,
                    role TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    commission REAL NOT NULL DEFAULT 0,
                    fees REAL NOT NULL DEFAULT 0,
                    FOREIGN KEY (trade_id) REFERENCES trades(trade_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS trade_annotations (
                    trade_id TEXT PRIMARY KEY,
                    setup_tags TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auto_tag_candidates (
                    trade_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    rationale TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (trade_id, tag)
                );

                CREATE TABLE IF NOT EXISTS tag_corrections (
                    correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    setup_tag TEXT NOT NULL,
                    source_trade_id TEXT NOT NULL DEFAULT '',
                    confidence_boost REAL NOT NULL DEFAULT 0.12,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS regimes (
                    trade_date TEXT PRIMARY KEY,
                    mid_term_regime TEXT NOT NULL DEFAULT '',
                    short_term_regime TEXT NOT NULL DEFAULT '',
                    intraday_regime TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_raw_exec_date ON raw_executions(trade_date);
                CREATE INDEX IF NOT EXISTS idx_raw_exec_symbol ON raw_executions(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(trade_date);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                ("schema_version", str(JOURNAL_SCHEMA_VERSION)),
            )

    def start_import_run(self, source: str) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO import_runs(source, status, started_at) VALUES(?, ?, ?)",
                (source, "RUNNING", _now_iso()),
            )
            return int(cursor.lastrowid)

    def finish_import_run(self, import_run_id: int, *, status: str, imported_executions: int, message: str = "") -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE import_runs
                SET status = ?, finished_at = ?, imported_executions = ?, message = ?
                WHERE import_run_id = ?
                """,
                (status, _now_iso(), int(imported_executions), str(message or ""), int(import_run_id)),
            )

    def upsert_accounts(self, broker: str, accounts: Iterable[dict[str, Any]]) -> int:
        rows = list(accounts)
        with self.connection() as conn:
            for account in rows:
                account_number = str(account.get("number") or account.get("accountNumber") or account.get("account_number") or "").strip()
                if not account_number:
                    continue
                conn.execute(
                    """
                    INSERT INTO accounts(
                        broker, account_number, account_label, account_type, currency, raw_json, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(broker, account_number) DO UPDATE SET
                        account_label = excluded.account_label,
                        account_type = excluded.account_type,
                        currency = excluded.currency,
                        raw_json = excluded.raw_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        str(broker or "").upper(),
                        account_number,
                        str(account.get("name") or account.get("description") or account.get("account_label") or account_number),
                        str(account.get("type") or account.get("accountType") or account.get("account_type") or ""),
                        str(account.get("currency") or ""),
                        _json_dumps(account),
                        _now_iso(),
                    ),
                )
        return len(rows)

    def upsert_executions(self, executions: Iterable[Any]) -> int:
        rows = [item.as_row() if hasattr(item, "as_row") else dict(item) for item in executions]
        with self.connection() as conn:
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO raw_executions(
                        execution_uid, broker, account_number, account_label, account_type, symbol,
                        security_type, currency, side, quantity, price, timestamp, trade_date,
                        commission, fees, gross_amount, net_amount, order_id, exchange_exec_id,
                        raw_json, imported_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(execution_uid) DO UPDATE SET
                        broker = excluded.broker,
                        account_number = excluded.account_number,
                        account_label = excluded.account_label,
                        account_type = excluded.account_type,
                        symbol = excluded.symbol,
                        security_type = excluded.security_type,
                        currency = excluded.currency,
                        side = excluded.side,
                        quantity = excluded.quantity,
                        price = excluded.price,
                        timestamp = excluded.timestamp,
                        trade_date = excluded.trade_date,
                        commission = excluded.commission,
                        fees = excluded.fees,
                        gross_amount = excluded.gross_amount,
                        net_amount = excluded.net_amount,
                        order_id = excluded.order_id,
                        exchange_exec_id = excluded.exchange_exec_id,
                        raw_json = excluded.raw_json,
                        imported_at = excluded.imported_at
                    """,
                    (
                        row.get("execution_uid"),
                        str(row.get("broker") or "").upper(),
                        str(row.get("account_number") or ""),
                        str(row.get("account_label") or ""),
                        str(row.get("account_type") or ""),
                        str(row.get("symbol") or "").upper(),
                        str(row.get("security_type") or "").upper(),
                        str(row.get("currency") or "USD").upper(),
                        str(row.get("side") or "").upper(),
                        _coerce_float(row.get("quantity")),
                        _coerce_float(row.get("price")),
                        str(row.get("timestamp") or _now_iso()),
                        _date_text(row.get("trade_date") or row.get("timestamp")),
                        abs(_coerce_float(row.get("commission"))),
                        abs(_coerce_float(row.get("fees"))),
                        row.get("gross_amount"),
                        row.get("net_amount"),
                        str(row.get("order_id") or ""),
                        str(row.get("exchange_exec_id") or ""),
                        str(row.get("raw_json") or "{}"),
                        _now_iso(),
                    ),
                )
        return len(rows)

    def _load_raw_executions(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM raw_executions
                ORDER BY broker, account_number, symbol, security_type, currency, timestamp, execution_uid
                """
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def rebuild_trades(self, *, refresh_tags: bool = True) -> int:
        executions = self._load_raw_executions()
        grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        for row in executions:
            key = (
                str(row.get("broker") or ""),
                str(row.get("account_number") or ""),
                str(row.get("symbol") or ""),
                str(row.get("security_type") or ""),
                str(row.get("currency") or ""),
            )
            grouped.setdefault(key, []).append(row)

        trade_payloads = []
        leg_payloads = []
        for key, rows in grouped.items():
            current = None
            sequence = 0
            for row in rows:
                signed_qty = _signed_quantity(row)
                if abs(signed_qty) <= EPSILON:
                    continue
                remaining_signed = signed_qty
                remaining_commission = abs(_coerce_float(row.get("commission")))
                remaining_fees = abs(_coerce_float(row.get("fees")))
                while abs(remaining_signed) > EPSILON:
                    if current is None:
                        sequence += 1
                        current = self._new_trade_state(key, row, remaining_signed, sequence)
                        self._add_open_quantity(current, row, remaining_signed, remaining_commission, remaining_fees)
                        break

                    pos = float(current["position_qty"])
                    if pos * remaining_signed > 0:
                        self._add_open_quantity(current, row, remaining_signed, remaining_commission, remaining_fees)
                        break

                    close_abs = min(abs(pos), abs(remaining_signed))
                    ratio = close_abs / abs(remaining_signed)
                    leg_signed = -math.copysign(close_abs, pos)
                    leg_commission = remaining_commission * ratio
                    leg_fees = remaining_fees * ratio
                    self._add_close_quantity(current, row, leg_signed, leg_commission, leg_fees)
                    remaining_abs = abs(remaining_signed) - close_abs
                    remaining_signed = math.copysign(remaining_abs, remaining_signed) if remaining_abs > EPSILON else 0.0
                    remaining_commission = max(0.0, remaining_commission - leg_commission)
                    remaining_fees = max(0.0, remaining_fees - leg_fees)

                    if abs(float(current["position_qty"])) <= EPSILON:
                        trade_payloads.append(self._finalize_trade_state(current))
                        leg_payloads.extend(current["legs"])
                        current = None

            if current is not None:
                trade_payloads.append(self._finalize_trade_state(current))
                leg_payloads.extend(current["legs"])

        with self.connection() as conn:
            conn.execute("DELETE FROM trade_legs")
            conn.execute("DELETE FROM auto_tag_candidates")
            conn.execute("DELETE FROM trades")
            for trade in trade_payloads:
                conn.execute(
                    """
                    INSERT INTO trades(
                        trade_id, broker, account_number, account_label, symbol, security_type, currency,
                        direction, status, opened_at, closed_at, trade_date, quantity_opened, quantity_closed,
                        average_entry_price, average_exit_price, gross_pnl, commission, fees, net_pnl,
                        pnl_usd, auto_tag_summary, tag_confidence, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade["trade_id"],
                        trade["broker"],
                        trade["account_number"],
                        trade["account_label"],
                        trade["symbol"],
                        trade["security_type"],
                        trade["currency"],
                        trade["direction"],
                        trade["status"],
                        trade["opened_at"],
                        trade["closed_at"],
                        trade["trade_date"],
                        trade["quantity_opened"],
                        trade["quantity_closed"],
                        trade["average_entry_price"],
                        trade["average_exit_price"],
                        trade["gross_pnl"],
                        trade["commission"],
                        trade["fees"],
                        trade["net_pnl"],
                        trade["pnl_usd"],
                        "",
                        None,
                        _now_iso(),
                    ),
                )
            for leg in leg_payloads:
                conn.execute(
                    """
                    INSERT INTO trade_legs(
                        trade_id, execution_uid, side, role, quantity, price, timestamp, commission, fees
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        leg["trade_id"],
                        leg["execution_uid"],
                        leg["side"],
                        leg["role"],
                        leg["quantity"],
                        leg["price"],
                        leg["timestamp"],
                        leg["commission"],
                        leg["fees"],
                    ),
                )

        if refresh_tags:
            self.refresh_auto_tags()
        return len(trade_payloads)

    def _new_trade_state(
        self,
        key: tuple[str, str, str, str, str],
        row: dict[str, Any],
        signed_qty: float,
        sequence: int,
    ) -> dict[str, Any]:
        direction = "LONG" if signed_qty > 0 else "SHORT"
        trade_id = _hash_id(*key, row.get("execution_uid"), direction, sequence)
        return {
            "trade_id": trade_id,
            "broker": key[0],
            "account_number": key[1],
            "symbol": key[2],
            "security_type": key[3],
            "currency": key[4] or "USD",
            "direction": direction,
            "account_label": str(row.get("account_label") or key[1]),
            "opened_at": str(row.get("timestamp") or ""),
            "last_at": str(row.get("timestamp") or ""),
            "position_qty": 0.0,
            "average_entry_price": 0.0,
            "entry_notional": 0.0,
            "exit_notional": 0.0,
            "quantity_opened": 0.0,
            "quantity_closed": 0.0,
            "gross_pnl": 0.0,
            "commission": 0.0,
            "fees": 0.0,
            "multiplier": _contract_multiplier(row),
            "legs": [],
        }

    def _add_trade_leg(
        self,
        trade: dict[str, Any],
        row: dict[str, Any],
        signed_qty: float,
        commission: float,
        fees: float,
        *,
        role: str,
    ) -> None:
        side = "BUY" if signed_qty > 0 else "SELL"
        qty = abs(float(signed_qty))
        trade["legs"].append(
            {
                "trade_id": trade["trade_id"],
                "execution_uid": row.get("execution_uid"),
                "side": side,
                "role": role,
                "quantity": qty,
                "price": _coerce_float(row.get("price")),
                "timestamp": str(row.get("timestamp") or ""),
                "commission": abs(float(commission)),
                "fees": abs(float(fees)),
            }
        )
        trade["commission"] = float(trade["commission"]) + abs(float(commission))
        trade["fees"] = float(trade["fees"]) + abs(float(fees))
        trade["last_at"] = str(row.get("timestamp") or trade["last_at"])

    def _add_open_quantity(self, trade: dict[str, Any], row: dict[str, Any], signed_qty: float, commission: float, fees: float) -> None:
        qty = abs(float(signed_qty))
        old_abs = abs(float(trade["position_qty"]))
        price = _coerce_float(row.get("price"))
        new_abs = old_abs + qty
        if new_abs > EPSILON:
            trade["average_entry_price"] = ((float(trade["average_entry_price"]) * old_abs) + (price * qty)) / new_abs
        trade["position_qty"] = float(trade["position_qty"]) + float(signed_qty)
        trade["quantity_opened"] = float(trade["quantity_opened"]) + qty
        trade["entry_notional"] = float(trade["entry_notional"]) + (price * qty)
        self._add_trade_leg(trade, row, signed_qty, commission, fees, role="OPEN" if old_abs <= EPSILON else "SCALE")

    def _add_close_quantity(self, trade: dict[str, Any], row: dict[str, Any], signed_qty: float, commission: float, fees: float) -> None:
        qty = abs(float(signed_qty))
        price = _coerce_float(row.get("price"))
        direction = 1.0 if str(trade["direction"]) == "LONG" else -1.0
        avg_entry = float(trade["average_entry_price"])
        multiplier = float(trade["multiplier"])
        trade["gross_pnl"] = float(trade["gross_pnl"]) + ((price - avg_entry) * qty * direction * multiplier)
        trade["position_qty"] = float(trade["position_qty"]) + float(signed_qty)
        trade["quantity_closed"] = float(trade["quantity_closed"]) + qty
        trade["exit_notional"] = float(trade["exit_notional"]) + (price * qty)
        self._add_trade_leg(trade, row, signed_qty, commission, fees, role="CLOSE")

    def _finalize_trade_state(self, trade: dict[str, Any]) -> dict[str, Any]:
        status = "CLOSED" if abs(float(trade["position_qty"])) <= EPSILON else "OPEN"
        closed_at = str(trade["last_at"] or "") if status == "CLOSED" else ""
        commission = float(trade["commission"])
        fees = float(trade["fees"])
        net_pnl = float(trade["gross_pnl"]) - commission - fees
        currency = str(trade["currency"] or "USD").upper()
        quantity_closed = float(trade["quantity_closed"])
        average_exit = (float(trade["exit_notional"]) / quantity_closed) if quantity_closed > EPSILON else 0.0
        return {
            "trade_id": trade["trade_id"],
            "broker": trade["broker"],
            "account_number": trade["account_number"],
            "account_label": trade["account_label"],
            "symbol": trade["symbol"],
            "security_type": trade["security_type"],
            "currency": currency,
            "direction": trade["direction"],
            "status": status,
            "opened_at": trade["opened_at"],
            "closed_at": closed_at,
            "trade_date": _date_text(closed_at or trade["opened_at"]),
            "quantity_opened": float(trade["quantity_opened"]),
            "quantity_closed": quantity_closed,
            "average_entry_price": float(trade["average_entry_price"]),
            "average_exit_price": average_exit,
            "gross_pnl": float(trade["gross_pnl"]),
            "commission": commission,
            "fees": fees,
            "net_pnl": net_pnl,
            "pnl_usd": net_pnl if currency == "USD" else None,
        }

    def list_trades(
        self,
        *,
        trade_date: str | date | None = None,
        broker: str | None = None,
        account: str | None = None,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if trade_date:
            date_value = _date_text(trade_date)
            clauses.append("(substr(opened_at, 1, 10) = ? OR substr(closed_at, 1, 10) = ? OR trade_date = ?)")
            params.extend([date_value, date_value, date_value])
        if broker and str(broker) != "All":
            clauses.append("broker = ?")
            params.append(str(broker).upper())
        if account and str(account) != "All":
            clauses.append("(account_number = ? OR account_label = ?)")
            params.extend([str(account), str(account)])
        if symbol:
            clauses.append("symbol = ?")
            params.append(str(symbol).upper())
        where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT t.*, COALESCE(a.setup_tags, '') AS setup_tags, COALESCE(a.notes, '') AS notes
                FROM trades t
                LEFT JOIN trade_annotations a ON a.trade_id = t.trade_id
                {where_sql}
                ORDER BY t.trade_date DESC, t.opened_at DESC, t.symbol
                """,
                params,
            ).fetchall()
        trades = [_row_to_dict(row) for row in rows]
        for trade in trades:
            regime = self.get_regime_for_date(_date_text(trade.get("opened_at") or trade.get("trade_date")))
            trade.update(regime)
            trade["display_tags"] = trade.get("setup_tags") or trade.get("auto_tag_summary") or ""
        return trades

    def list_trade_legs(self, trade_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT l.*, e.broker, e.account_number, e.symbol, e.security_type, e.currency
                FROM trade_legs l
                LEFT JOIN raw_executions e ON e.execution_uid = l.execution_uid
                WHERE l.trade_id = ?
                ORDER BY l.timestamp, l.leg_id
                """,
                (trade_id,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_trade(self, trade_id: str) -> dict[str, Any] | None:
        rows = self.list_trades()
        for row in rows:
            if row.get("trade_id") == trade_id:
                return row
        return None

    def save_trade_annotation(self, trade_id: str, *, setup_tags: str, notes: str) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO trade_annotations(trade_id, setup_tags, notes, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    setup_tags = excluded.setup_tags,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (trade_id, str(setup_tags or "").strip(), str(notes or "").strip(), _now_iso()),
            )

    def record_tag_corrections(self, trade: dict[str, Any], tags: str) -> None:
        symbol = str(trade.get("symbol") or "").strip().upper()
        if not symbol:
            return
        parsed_tags = [part.strip() for part in str(tags or "").replace(",", ";").split(";") if part.strip()]
        if not parsed_tags:
            return
        with self.connection() as conn:
            for tag in parsed_tags:
                conn.execute(
                    """
                    INSERT INTO tag_corrections(symbol, setup_tag, source_trade_id, confidence_boost, created_at)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (symbol, tag, str(trade.get("trade_id") or ""), 0.12, _now_iso()),
                )

    def list_tag_corrections(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT symbol, setup_tag, source_trade_id, confidence_boost, created_at FROM tag_corrections"
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def refresh_auto_tags(self, tagger: AutoTagger | None = None) -> None:
        tagger = tagger or AutoTagger()
        corrections = self.list_tag_corrections()
        with self.connection() as conn:
            trade_rows = conn.execute("SELECT * FROM trades ORDER BY opened_at").fetchall()
            for row in trade_rows:
                trade = _row_to_dict(row)
                suggestions = tagger.suggest_for_trade(trade, corrections=corrections)
                top_summary = "; ".join(item["tag"] for item in suggestions[:3])
                top_confidence = suggestions[0]["confidence"] if suggestions else None
                conn.execute("DELETE FROM auto_tag_candidates WHERE trade_id = ?", (trade["trade_id"],))
                for item in suggestions:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO auto_tag_candidates(
                            trade_id, tag, confidence, source, rationale, created_at
                        ) VALUES(?, ?, ?, ?, ?, ?)
                        """,
                        (
                            trade["trade_id"],
                            item["tag"],
                            float(item["confidence"]),
                            item.get("source", ""),
                            item.get("rationale", ""),
                            _now_iso(),
                        ),
                    )
                conn.execute(
                    "UPDATE trades SET auto_tag_summary = ?, tag_confidence = ?, updated_at = ? WHERE trade_id = ?",
                    (top_summary, top_confidence, _now_iso(), trade["trade_id"]),
                )

    def list_auto_tag_candidates(self, trade_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM auto_tag_candidates
                WHERE trade_id = ?
                ORDER BY confidence DESC, tag
                """,
                (trade_id,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def upsert_regime(
        self,
        trade_date: str | date,
        *,
        mid_term_regime: str = "",
        short_term_regime: str = "",
        intraday_regime: str = "",
        notes: str = "",
    ) -> None:
        date_value = _date_text(trade_date)
        with self.connection() as conn:
            existing = conn.execute("SELECT * FROM regimes WHERE trade_date = ?", (date_value,)).fetchone()
            row = _row_to_dict(existing) if existing else {}
            conn.execute(
                """
                INSERT INTO regimes(
                    trade_date, mid_term_regime, short_term_regime, intraday_regime, notes, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_date) DO UPDATE SET
                    mid_term_regime = excluded.mid_term_regime,
                    short_term_regime = excluded.short_term_regime,
                    intraday_regime = excluded.intraday_regime,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (
                    date_value,
                    str(mid_term_regime if mid_term_regime != "" else row.get("mid_term_regime", "")).strip(),
                    str(short_term_regime if short_term_regime != "" else row.get("short_term_regime", "")).strip(),
                    str(intraday_regime if intraday_regime != "" else row.get("intraday_regime", "")).strip(),
                    str(notes if notes != "" else row.get("notes", "")).strip(),
                    _now_iso(),
                ),
            )

    def get_regime_for_date(self, trade_date: str | date) -> dict[str, str]:
        date_value = _date_text(trade_date)
        with self.connection() as conn:
            exact = conn.execute("SELECT * FROM regimes WHERE trade_date = ?", (date_value,)).fetchone()
            carry = conn.execute(
                """
                SELECT * FROM regimes
                WHERE trade_date <= ?
                  AND (mid_term_regime != '' OR short_term_regime != '')
                ORDER BY trade_date DESC
                LIMIT 1
                """,
                (date_value,),
            ).fetchone()
        exact_row = _row_to_dict(exact) if exact else {}
        carry_row = _row_to_dict(carry) if carry else {}
        return {
            "mid_term_regime": str(exact_row.get("mid_term_regime") or carry_row.get("mid_term_regime") or ""),
            "short_term_regime": str(exact_row.get("short_term_regime") or carry_row.get("short_term_regime") or ""),
            "intraday_regime": str(exact_row.get("intraday_regime") or ""),
            "regime_notes": str(exact_row.get("notes") or ""),
        }

    def list_import_runs(self, limit: int = 25) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM import_runs
                ORDER BY import_run_id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def distinct_values(self, column: str) -> list[str]:
        if column not in {"broker", "account_number", "account_label", "symbol"}:
            return []
        with self.connection() as conn:
            rows = conn.execute(f"SELECT DISTINCT {column} AS value FROM trades ORDER BY value").fetchall()
        return [str(row["value"]) for row in rows if str(row["value"] or "").strip()]

    def export_trades_csv(self, path: Path | None = None) -> Path:
        target = Path(path) if path else JOURNAL_EXPORT_DIR / f"journal_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self.list_trades()
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return target
