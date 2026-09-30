"""Read-only journal helpers shared by the mirror and tilt packs. Qt-free, never builds ``JournalStore``.

The journal is opened ``mode=ro``. Times are parsed as ISO 8601 (fractional seconds and
offsets included); a naive stamp reads as New York. Option legs of one account and
underlying opened within :data:`SPREAD_WINDOW` of the first leg are one spread unit. R is
net PnL over the planned risk (``planned_risk``, else |entry - stop| x qty x multiplier);
no stop and no risk = R unknown, never a guess.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
#: Option legs opened this close together in one account and underlying are one spread.
SPREAD_WINDOW = timedelta(minutes=3)
OPTION_MULTIPLIER = 100.0
_OPTION_TYPES = ("OPT", "OPTION", "OPTIONS", "EQUITYOPTION", "FOP")
#: Questrade's option spelling (``AAOI18Jun26P120.00``) and the OCC one (``AEHR261002P00085000``).
_OPTION_SYMBOL = re.compile(
    r"^[A-Z][A-Z.]{0,5}?(0?[1-9]|[12][0-9]|3[01])(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
    r"[0-9]{2}[CP][0-9]+(\.[0-9]+)?$|^[A-Z][A-Z.]{0,5}[0-9]{6}[CP][0-9]{8}$"
)
_ROOT = re.compile(r"^([A-Z][A-Z.]*?)(?=[0-9])")


def connect_ro(path: Path | str) -> sqlite3.Connection | None:
    target = Path(path)
    if not target.exists():
        return None
    conn = sqlite3.connect(f"{target.as_uri()}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def parse_time(value: Any) -> datetime | None:
    """An ISO 8601 stamp as an aware New York datetime; naive = New York; unreadable = None."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (moment if moment.tzinfo else moment.replace(tzinfo=ET)).astimezone(ET)


def num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def is_option(trade: Mapping[str, Any]) -> bool:
    kind = re.sub(r"[\s_\-]", "", str(trade.get("security_type") or "")).upper()
    if kind in _OPTION_TYPES:
        return True
    symbol = re.sub(r"\s+", "", str(trade.get("symbol") or "").upper())
    return _OPTION_SYMBOL.fullmatch(symbol) is not None


def underlying(symbol: Any) -> str:
    text = re.sub(r"\s+", "", str(symbol or "").upper())
    found = _ROOT.match(text)
    return found.group(1) if found else text


def pnl(trade: Mapping[str, Any]) -> float | None:
    """Net PnL in USD when the journal has it, else the trade's own net."""
    value = num(trade.get("net_pnl_usd"))
    return value if value is not None else num(trade.get("net_pnl"))


# ---------------------------------------------------------------- readers
def read_trades(path: Path | str, *, since: str = "") -> list[dict[str, Any]]:
    """Trades opened or closed on/after ``since`` (YYYY-MM-DD; blank = all), with their plan fields."""
    conn = connect_ro(path)
    if conn is None:
        return []
    try:
        if not _has_table(conn, "trades"):
            return []
        have = _columns(conn, "trades")
        wanted = ("trade_id", "account_number", "account_label", "symbol", "security_type", "direction", "status",
                  "opened_at", "closed_at", "quantity_opened", "quantity_closed", "average_entry_price",
                  "average_exit_price", "net_pnl", "net_pnl_usd")
        cols = ", ".join(f"t.{c}" if c in have else f"NULL AS {c}" for c in wanted)
        plan = "NULL AS planned_stop, NULL AS planned_risk"
        join = ""
        if _has_table(conn, "trade_annotations"):
            ann = _columns(conn, "trade_annotations")
            plan = ", ".join(f"a.{c}" if c in ann else f"NULL AS {c}" for c in ("planned_stop", "planned_risk"))
            join = "LEFT JOIN trade_annotations a ON a.trade_id = t.trade_id"
        sql = f"SELECT {cols}, {plan} FROM trades t {join}"
        params: tuple[Any, ...] = ()
        if since:
            sql += " WHERE substr(t.opened_at, 1, 10) >= ? OR substr(COALESCE(t.closed_at, ''), 1, 10) >= ?"
            params = (since, since)
        return [dict(row) for row in conn.execute(sql + " ORDER BY t.opened_at, t.trade_id", params)]
    finally:
        conn.close()


def read_legs(path: Path | str, *, since: str = "") -> list[dict[str, Any]]:
    """Legs (with their trade's symbol, direction, account and type) stamped on/after ``since``."""
    conn = connect_ro(path)
    if conn is None:
        return []
    try:
        if not (_has_table(conn, "trade_legs") and _has_table(conn, "trades")):
            return []
        have = _columns(conn, "trades")
        extra = ", ".join(f"t.{c}" if c in have else f"NULL AS {c}"
                          for c in ("symbol", "direction", "account_number", "security_type"))
        sql = (f"SELECT l.leg_id, l.trade_id, l.side, l.role, l.quantity, l.price, l.timestamp, {extra} "
               "FROM trade_legs l JOIN trades t ON t.trade_id = l.trade_id")
        params: tuple[Any, ...] = ()
        if since:
            sql += " WHERE substr(l.timestamp, 1, 10) >= ?"
            params = (since,)
        return [dict(row) for row in conn.execute(sql + " ORDER BY l.timestamp, l.leg_id", params)]
    finally:
        conn.close()


def read_accounts(path: Path | str) -> list[dict[str, Any]]:
    from mentor_packs.book_pack import read_accounts as _accounts

    return _accounts(Path(path))


def read_regime_rows(path: Path | str) -> list[dict[str, Any]]:
    conn = connect_ro(path)
    if conn is None:
        return []
    try:
        if not _has_table(conn, "structural_regime"):
            return []
        return [dict(row) for row in conn.execute("SELECT * FROM structural_regime ORDER BY segment_id")]
    finally:
        conn.close()


def leg_signature(path: Path | str, day: str) -> tuple[int, int, int] | None:
    """(max leg id, legs on ``day``, trades closed on ``day``): unchanged = nothing new to read."""
    conn = connect_ro(path)
    if conn is None:
        return None
    try:
        if not (_has_table(conn, "trade_legs") and _has_table(conn, "trades")):
            return None
        top = conn.execute("SELECT COALESCE(MAX(leg_id), 0) FROM trade_legs").fetchone()[0]
        legs = conn.execute("SELECT COUNT(*) FROM trade_legs WHERE substr(timestamp, 1, 10) = ?", (day,)).fetchone()[0]
        closed = conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'CLOSED' AND substr(closed_at, 1, 10) = ?",
                              (day,)).fetchone()[0]
        return int(top or 0), int(legs or 0), int(closed or 0)
    finally:
        conn.close()


# ---------------------------------------------------------------- units (spreads paired)
@dataclass
class Unit:
    """One decision: a trade, or option legs opened together (a spread)."""

    trades: list[dict[str, Any]]
    kind: str  # "day" | "swing" | "option"
    opened: datetime | None
    closed: datetime | None
    pnl: float | None
    r: float | None
    account_number: str = ""
    symbol: str = ""
    ids: list[str] = field(default_factory=list)


def _risk(trade: Mapping[str, Any]) -> float | None:
    planned = num(trade.get("planned_risk"))
    if planned is not None and planned > 0:
        return planned
    stop, entry, qty = num(trade.get("planned_stop")), num(trade.get("average_entry_price")), num(trade.get("quantity_opened"))
    if None in (stop, entry, qty):
        return None
    risk = abs(entry - stop) * qty * (OPTION_MULTIPLIER if is_option(trade) else 1.0)
    return risk if risk > 0 else None


def _closed(trade: Mapping[str, Any]) -> bool:
    return str(trade.get("status") or "").upper() == "CLOSED" and parse_time(trade.get("closed_at")) is not None


def units(trades: Iterable[Mapping[str, Any]]) -> list[Unit]:
    """Closed decisions: stock trades one each; option trades grouped into spreads. Open ones are left out."""
    stock: list[Unit] = []
    options: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw in trades:
        trade = dict(raw)
        if is_option(trade):
            options.setdefault((str(trade.get("account_number") or ""), underlying(trade.get("symbol"))), []).append(trade)
            continue
        if not _closed(trade):
            continue
        opened, closed = parse_time(trade.get("opened_at")), parse_time(trade.get("closed_at"))
        kind = "day" if opened is not None and closed is not None and opened.date() == closed.date() else "swing"
        value, risk = pnl(trade), _risk(trade)
        stock.append(Unit([trade], kind, opened, closed, value,
                          value / risk if value is not None and risk else None,
                          str(trade.get("account_number") or ""), str(trade.get("symbol") or "").upper(),
                          [str(trade.get("trade_id"))]))
    for (account, root), group in options.items():
        group.sort(key=lambda t: parse_time(t.get("opened_at")) or datetime.max.replace(tzinfo=ET))
        clusters: list[list[dict[str, Any]]] = []
        for trade in group:
            opened = parse_time(trade.get("opened_at"))
            # Within SPREAD_WINDOW of the cluster's FIRST leg; legs are never chained one to the next.
            first = parse_time(clusters[-1][0].get("opened_at")) if clusters else None
            if clusters and opened is not None and first is not None and opened - first <= SPREAD_WINDOW:
                clusters[-1].append(trade)
            else:
                clusters.append([trade])
        for cluster in clusters:
            if not all(_closed(t) for t in cluster):
                continue
            values = [pnl(t) for t in cluster]
            value = None if any(v is None for v in values) else float(sum(values))
            risks = [_risk(t) for t in cluster]
            # A spread's risk is only known when every leg carries one; one leg alone never stands for the spread.
            risk = float(sum(risks)) if len(cluster) == 1 and risks[0] else None
            stock.append(Unit(cluster, "option", parse_time(cluster[0].get("opened_at")),
                              max(parse_time(t.get("closed_at")) for t in cluster), value,
                              value / risk if value is not None and risk else None, account, root,
                              [str(t.get("trade_id")) for t in cluster]))
    stock.sort(key=lambda u: (u.closed or datetime.min.replace(tzinfo=ET), u.ids))
    return stock
