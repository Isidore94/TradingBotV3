"""Book pack: the trader's open book by account, with account/tax hints. Read-only.

One source per pack, named in ``book:source``: the Questrade snapshot when it is fresh
(at most 15 min old; the app's ``/book`` fetch keeps it in its own chat store), else the
journal's open trades (``mode=ro``) with the reason Questrade was not used. The two are
never mixed. Rows: ``book:asof``, ``book:source``, ``book:acct:<N>`` (label, tax class,
position count, cash when known), ``book:pos:<ACCT>:<SYM>`` (side, qty, avg, market
value, $ at risk when the journal has a planned stop), ``book:side:<LONG|SHORT>``,
``book:industry:<name>`` (top 3 by count and every cluster of 2+), ``book:hint:<k>``
(deterministic: no shorts in registered accounts, industry clusters, room per account
only when ``mentor_max_positions_per_account`` is set). Never sizes, orders or writes.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.registry import Pack, make_pack

NAME = "book_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "The trader's open book: accounts (TFSA/RRSP are registered and cannot hold shorts), open "
            "positions with side, size, average and market value, exposure by side and industry, and "
            "account hints. Says whether it came from Questrade or the journal. Never sizes or orders."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

PT = ZoneInfo("America/Los_Angeles")
MAX_POSITIONS_SETTING = "mentor_max_positions_per_account"
SNAPSHOT_KEY = "book:snapshot"
STATUS_KEY = "book:last_error"
TOP_INDUSTRIES = 3
#: Tax-free / tax-deferred account words (``journal_analytics.REGISTERED_ACCOUNT_WORDS``).
TAX_FREE_WORDS = ("TFSA", "FHSA")
TAX_DEFERRED_WORDS = ("RRSP", "RRIF", "RESP", "LIRA", "LIF", "RDSP", "SPOUSAL")


@dataclass(frozen=True)
class Sources:
    """Where each section reads from; tests pass fixtures, the app uses :func:`live_sources`."""

    #: The last stored Questrade snapshot as a dict (any age), or None.
    snapshot: Callable[[], Mapping[str, Any] | None]
    #: ``{reason, at_utc}`` of the last failed Questrade fetch, or None.
    status: Callable[[], Mapping[str, Any] | None]
    open_trades: Callable[[], list[Mapping[str, Any]]]
    #: The journal's ``accounts`` table rows (label, type, trader-owned tax_status).
    accounts: Callable[[], list[Mapping[str, Any]]]
    industry_map: Callable[[], Mapping[str, Mapping[str, Any]]]
    max_positions: Callable[[], Any] = lambda: None
    now: datetime | None = field(default=None, compare=False)


# ---------------------------------------------------------------- live readers (mode=ro)
def _connect_ro(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _state(path: Path, key: str) -> str | None:
    conn = _connect_ro(path)
    if conn is None:
        return None
    try:
        found = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        return str(found["value"]) if found else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _json_state(path: Path, key: str) -> dict[str, Any] | None:
    try:
        value = json.loads(_state(path, key) or "null")
    except ValueError:
        return None
    return value if isinstance(value, dict) and value else None


def db_snapshot_reader(path: Path | str) -> Callable[[], dict[str, Any] | None]:
    return lambda: _json_state(Path(path), SNAPSHOT_KEY)


def db_status_reader(path: Path | str) -> Callable[[], dict[str, Any] | None]:
    return lambda: _json_state(Path(path), STATUS_KEY)


def read_accounts(path: Path) -> list[dict[str, Any]]:
    """The journal's accounts (``mode=ro``); a missing file or table is none."""
    conn = _connect_ro(Path(path))
    if conn is None:
        return []
    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='accounts'").fetchone() is None:
            return []
        return [{key: row[key] for key in row.keys() if key != "raw_json"}
                for row in conn.execute("SELECT * FROM accounts ORDER BY broker, account_number")]
    finally:
        conn.close()


def live_sources() -> Sources:
    from project_paths import JOURNAL_DB_FILE, MENTOR_CHAT_DB_FILE, get_local_setting

    from mentor_packs import gate_pack

    return Sources(
        snapshot=db_snapshot_reader(MENTOR_CHAT_DB_FILE),
        status=db_status_reader(MENTOR_CHAT_DB_FILE),
        open_trades=lambda: gate_pack.read_open_trades(Path(JOURNAL_DB_FILE)),
        accounts=lambda: read_accounts(Path(JOURNAL_DB_FILE)),
        industry_map=gate_pack._live_industry_map,
        max_positions=lambda: get_local_setting(MAX_POSITIONS_SETTING, None),
    )


# ---------------------------------------------------------------- helpers
def _sym(value: Any) -> str:
    return str(value or "").strip().upper()


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _money(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"-${-value:,.2f}" if value < 0 else f"${value:,.2f}"


def _qty(value: float) -> str:
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _id_part(value: Any) -> str:
    """A text safe inside a citation id (``[A-Za-z0-9_.:-]``)."""
    return re.sub(r"[^A-Za-z0-9_.\-]+", "-", str(value or "").strip()).strip("-") or "unknown"


def _pt(value: str | datetime) -> str:
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    moment = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    return f"{moment.astimezone(PT):%a %m-%d %H:%M} PT"


def _now(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.astimezone()


def tax_class(account: Mapping[str, Any]) -> str:
    """``tax_free`` / ``tax_deferred`` / ``margin`` / ``cash`` / ``taxable`` / ``unknown``.

    The trader's own tax status in the journal wins; then the account type words.
    """
    status = str(account.get("tax_status") or "").strip().upper()
    words = f"{account.get('account_type') or ''} {account.get('account_label') or ''}".upper()
    if status == "TAX_FREE":
        return "tax_free"
    if status == "TAX_DEFERRED":
        return "tax_deferred"
    if any(word in words for word in TAX_FREE_WORDS):
        return "tax_free"
    if any(word in words for word in TAX_DEFERRED_WORDS):
        return "tax_deferred"
    if "MARGIN" in words:
        return "margin"
    if re.search(r"\bCASH\b", words):
        return "cash"
    return "taxable" if status == "TAXABLE" else "unknown"


CLASS_TEXT = {
    "tax_free": "registered, tax-free, no shorts",
    "tax_deferred": "registered, tax-deferred, no shorts",
    "margin": "margin",
    "cash": "cash account, no shorts",
    "taxable": "non-registered",
    "unknown": "account type unknown",
}
NO_SHORTS = frozenset({"tax_free", "tax_deferred", "cash"})


def is_registered(klass: str) -> bool:
    return klass in ("tax_free", "tax_deferred")


# ---------------------------------------------------------------- the book, from one source
def _questrade_book(snap: Mapping[str, Any], journal_accounts: list[Mapping[str, Any]]) -> tuple[list, list]:
    by_number = {str(a.get("account_number") or ""): a for a in journal_accounts}
    accounts = []
    for account in snap.get("accounts") or ():
        number = str(account.get("account_number") or "")
        own = by_number.get(number) or {}
        accounts.append({**dict(account), "tax_status": own.get("tax_status") or ""})
    return accounts, [dict(p) for p in snap.get("positions") or ()]


def _journal_book(trades: Iterable[Mapping[str, Any]], journal_accounts: list[Mapping[str, Any]]) -> tuple[list, list]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for trade in trades:
        sym = _sym(trade.get("symbol"))
        side = str(trade.get("direction") or "").strip().upper()
        side = {"BUY": "LONG", "SELL": "SHORT"}.get(side, side)
        qty = (_num(trade.get("quantity_opened")) or 0.0) - (_num(trade.get("quantity_closed")) or 0.0)
        if not sym or qty <= 0:
            continue
        acct = str(trade.get("account_number") or "") or "journal"
        key = (acct, sym, side)
        entry = _num(trade.get("average_entry_price"))
        row = grouped.setdefault(key, {"account_number": acct, "account_label": str(trade.get("account_label") or ""),
                                       "symbol": sym, "side": side, "open_qty": 0.0, "cost": 0.0,
                                       "avg_known": True, "market_value": None})
        row["open_qty"] += qty
        if entry is None:
            row["avg_known"] = False
        else:
            row["cost"] += entry * qty
    positions = []
    for row in grouped.values():
        avg = row.pop("cost") / row["open_qty"] if row.pop("avg_known") and row["open_qty"] else None
        positions.append({**row, "avg_price": avg})
    known = {str(a.get("account_number") or ""): dict(a) for a in journal_accounts}
    for pos in positions:
        known.setdefault(pos["account_number"], {"account_number": pos["account_number"],
                                                 "account_label": pos["account_label"] or pos["account_number"]})
    accounts = [{**a, "cash": None, "cash_known": False} for a in known.values() if a.get("account_number")]
    return accounts, positions


def _stops(trades: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], float]:
    """(SYMBOL, SIDE) -> the planned stop of the latest open journal trade that has one."""
    out: dict[tuple[str, str], float] = {}
    for trade in trades:  # oldest first, so the latest wins
        stop = _num(trade.get("planned_stop"))
        side = {"BUY": "LONG", "SELL": "SHORT"}.get(str(trade.get("direction") or "").upper(),
                                                    str(trade.get("direction") or "").upper())
        if stop is not None:
            out[(_sym(trade.get("symbol")), side)] = stop
    return out


def _setting_int(value: Any) -> int | None:
    number = _num(value)
    return int(number) if number is not None and number >= 1 else None


@dataclass
class Book:
    """The parsed book the rows are built from (also the gate pack's book section)."""

    source: str  # "questrade" | "journal"
    source_text: str
    accounts: list[dict[str, Any]]
    positions: list[dict[str, Any]]
    industries: dict[str, str]
    max_positions: int | None
    asof: str


def load_book(src: Sources, now: datetime | None = None) -> Book:
    """Choose the source (fresh Questrade, else the journal) and read it. File/DB reads: a worker only."""
    from questrade_positions import BookSnapshot

    moment = _now(now or src.now)
    try:
        journal_accounts = list(src.accounts() or ())
    except Exception:  # noqa: BLE001 - the tax label is optional; unreadable = the type words
        journal_accounts = []
    raw = src.snapshot()
    snap = BookSnapshot.from_dict(raw) if raw else None
    trades = list(src.open_trades() or ())
    if snap is not None and snap.is_fresh(moment):
        accounts, positions = _questrade_book(snap.as_dict(), journal_accounts)
        source, asof = "questrade", snap.fetched_utc
        text = f"Source: Questrade positions at {_pt(asof)}"
    else:
        status = src.status() or {}
        if snap is not None and status.get("at_utc", "") <= snap.fetched_utc:
            age = snap.age(moment)
            why = f"last snapshot {int(age.total_seconds() // 60)} min old" if age is not None else "snapshot unreadable"
        elif status.get("reason"):
            why = str(status["reason"])
        else:
            why = "not fetched yet"
        accounts, positions = _journal_book(trades, journal_accounts)
        source, asof = "journal", moment.astimezone(timezone.utc).isoformat(timespec="seconds")
        text = f"Source: journal open trades (Questrade unavailable: {why})"
    stops = _stops(trades)
    for pos in positions:
        stop = stops.get((_sym(pos.get("symbol")), str(pos.get("side") or "")))
        avg, qty = _num(pos.get("avg_price")), _num(pos.get("open_qty"))
        pos["stop"] = stop
        pos["at_risk"] = round(abs(avg - stop) * qty, 2) if None not in (stop, avg, qty) else None
    try:
        imap = src.industry_map() or {}
    except Exception:  # noqa: BLE001 - industry is a label; unreadable = unknown
        imap = {}
    industries = {_sym(p.get("symbol")): str((imap.get(_sym(p.get("symbol"))) or {}).get("industry") or "").strip()
                  for p in positions}
    try:
        max_positions = _setting_int(src.max_positions())
    except Exception:  # noqa: BLE001 - no setting = no room hint, never a guess
        max_positions = None
    for account in accounts:
        account["class"] = tax_class(account)
        account["count"] = sum(1 for p in positions if str(p.get("account_number")) == str(account["account_number"]))
    return Book(source, text, accounts, positions, industries, max_positions, asof)


# ---------------------------------------------------------------- rows
def _label(account: Mapping[str, Any]) -> str:
    return str(account.get("account_label") or account.get("account_number") or "account")


def _cash_text(account: Mapping[str, Any], source: str) -> str:
    cash = account.get("cash")
    if account.get("cash_known") and isinstance(cash, Mapping):
        return "cash " + ", ".join(f"{cur} {value:,.2f}" for cur, value in sorted(cash.items()))
    return "cash unknown" + (" (the journal has no cash)" if source == "journal" else "")


def account_rows(book: Book) -> list[dict[str, Any]]:
    rows = []
    for account in book.accounts:
        klass = account["class"]
        rows.append({"id": f"book:acct:{_id_part(account['account_number'])}", "kind": "account",
                     "account_number": str(account["account_number"]), "label": _label(account),
                     "class": klass, "count": account["count"],
                     "text": (f"Account {_label(account)} ({account.get('account_type') or 'type unknown'}): "
                              f"{CLASS_TEXT[klass]}; {account['count']} open position(s); {_cash_text(account, book.source)}")})
    return rows


def position_rows(book: Book) -> list[dict[str, Any]]:
    rows, seen = [], set()
    labels = {str(a["account_number"]): _label(a) for a in book.accounts}
    for pos in sorted(book.positions, key=lambda p: (str(p.get("account_number")), _sym(p.get("symbol")),
                                                      str(p.get("side")))):
        acct, sym, side = str(pos.get("account_number") or ""), _sym(pos.get("symbol")), str(pos.get("side") or "?")
        row_id = f"book:pos:{_id_part(acct)}:{_id_part(sym)}"
        if row_id in seen:
            row_id += f":{side}"
        seen.add(row_id)
        qty, avg, value = _num(pos.get("open_qty")) or 0.0, _num(pos.get("avg_price")), _num(pos.get("market_value"))
        risk = (f"$ at risk {_money(pos['at_risk'])} (stop {pos['stop']:g})" if pos.get("at_risk") is not None
                else "stop unknown")
        rows.append({"id": row_id, "kind": "position", "account_number": acct, "symbol": sym, "side": side,
                     "qty": qty, "avg_price": avg, "market_value": value, "at_risk": pos.get("at_risk"),
                     "industry": book.industries.get(sym, ""),
                     "text": (f"{side} {sym} {_qty(qty)} @ {'unknown' if avg is None else f'{avg:,.2f}'} in "
                              f"{labels.get(acct, acct)}; market value {_money(value)}; {risk}")})
    return rows


def exposure_rows(book: Book) -> list[dict[str, Any]]:
    rows = []
    for side in ("LONG", "SHORT"):
        mine = [p for p in book.positions if p.get("side") == side]
        values = [_num(p.get("market_value")) for p in mine]
        known = [abs(v) for v in values if v is not None]
        gross = float(sum(known)) if len(known) == len(mine) else None
        text = f"{side}: {len(mine)} position(s), gross market value " + (
            _money(gross) if gross is not None else f"unknown ({len(mine) - len(known)} without a market value)")
        rows.append({"id": f"book:side:{side}", "kind": "side", "side": side, "count": len(mine),
                     "gross": gross, "text": text})
    names: dict[str, set[str]] = {}
    for pos in book.positions:
        sym = _sym(pos.get("symbol"))
        industry = book.industries.get(sym) or ""
        if industry:
            names.setdefault(industry, set()).add(sym)
    counts = Counter({industry: len(syms) for industry, syms in names.items()})
    top = [industry for industry, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_INDUSTRIES]]
    clusters = sorted(industry for industry, n in counts.items() if n >= 2)
    for industry in top + [c for c in clusters if c not in top]:
        syms = sorted(names[industry])
        rows.append({"id": f"book:industry:{_id_part(industry)}", "kind": "industry", "industry": industry,
                     "count": len(syms), "names": syms, "cluster": len(syms) >= 2,
                     "text": f"Industry {industry}: {len(syms)} open name(s) ({', '.join(syms)})"})
    unknown = sorted(sym for sym, industry in book.industries.items() if not industry)
    if unknown:
        rows.append({"id": "book:industry:unknown", "kind": "industry_unknown", "names": unknown,
                     "text": f"Industry unknown for {', '.join(unknown)}"})
    return rows


def hint_rows(book: Book) -> list[dict[str, Any]]:
    rows = []
    registered = [a for a in book.accounts if is_registered(a["class"])]
    if registered:
        labels = ", ".join(_label(a) for a in registered)
        rows.append({"id": "book:hint:no_shorts_registered", "kind": "hint",
                     "accounts": [str(a["account_number"]) for a in registered],
                     "text": f"Short candidates cannot go in {labels} (registered accounts hold no shorts)"})
    names: dict[str, set[str]] = {}
    for pos in book.positions:
        industry = book.industries.get(_sym(pos.get("symbol"))) or ""
        if industry:
            names.setdefault(industry, set()).add(_sym(pos.get("symbol")))
    for industry in sorted(names):
        if len(names[industry]) >= 2:
            syms = sorted(names[industry])
            rows.append({"id": f"book:hint:cluster:{_id_part(industry)}", "kind": "hint", "industry": industry,
                         "text": f"{len(syms)} open names in {industry} ({', '.join(syms)})"})
    if book.max_positions is not None:
        for account in book.accounts:
            room = book.max_positions - account["count"]
            text = (f"{_label(account)} has room: {room} position(s) (max {book.max_positions} per account)"
                    if room > 0 else f"{_label(account)} is full: {account['count']} of {book.max_positions} positions")
            rows.append({"id": f"book:hint:room:{_id_part(account['account_number'])}", "kind": "hint_room",
                         "account_number": str(account["account_number"]), "room": max(0, room),
                         "class": account["class"], "text": text})
    return rows


def short_hint(book: Book) -> dict[str, Any] | None:
    """For a short request: which accounts cannot take it and, when room is set, whether any that can has room.

    None when every account can hold a short (or there are none): nothing to say.
    """
    blocked = [a for a in book.accounts if a["class"] in NO_SHORTS]
    if not blocked:
        return None
    names = ", ".join(_label(a) for a in blocked)
    row: dict[str, Any] = {"id": "book:hint:short_account", "kind": "hint_short",
                           "blocked": [str(a["account_number"]) for a in blocked], "no_room_for_short": False}
    if book.max_positions is None:
        row["text"] = f"This short cannot go in {names} (no shorts there)"
        return row
    room = [a for a in book.accounts if book.max_positions - a["count"] > 0]
    margin = [a for a in room if a["class"] == "margin"]
    unknown = [a for a in room if a["class"] not in NO_SHORTS and a["class"] != "margin"]
    if not room:
        row["no_room_for_short"] = True
        row["text"] = f"No account has room (max {book.max_positions} per account); {names} cannot hold a short anyway"
    elif margin:
        row["text"] = f"{', '.join(_label(a) for a in margin)} has room for this short; {names} cannot hold one"
    elif unknown:
        row["text"] = (f"{', '.join(_label(a) for a in unknown)} has room, but whether it can hold a short is "
                       f"unknown; {names} cannot hold one")
    else:
        row["no_room_for_short"] = True
        row["text"] = (f"Only {', '.join(_label(a) for a in room)} has room, and it cannot hold a short: "
                       f"no account with room can take this short")
    return row


def book_rows(book: Book) -> list[dict[str, Any]]:
    rows = [{"id": "book:asof", "kind": "asof", "text": f"Book as of {_pt(book.asof)}"},
            {"id": "book:source", "kind": "source", "source": book.source, "text": book.source_text}]
    rows += account_rows(book)
    positions = position_rows(book)
    rows += positions or [{"id": "book:pos:none", "kind": "position_none", "text": "No open positions"}]
    rows += exposure_rows(book)
    rows += hint_rows(book)
    return rows


def build(*, now: datetime | None = None, sources: Sources | None = None) -> Pack:
    """Build the book pack. File and DB reads: call it on a worker."""
    src = sources or live_sources()
    return make_pack(NAME, book_rows(load_book(src, now)))


# ---------------------------------------------------------------- fixture
FIXTURE_NOW = datetime(2026, 9, 29, 7, 5, tzinfo=PT)
FIXTURE_INDUSTRIES = {
    "NVDA": {"industry": "Semiconductors"}, "AMD": {"industry": "Semiconductors"},
    "AVGO": {"industry": "Semiconductors"}, "TSLA": {"industry": "Auto Manufacturers"},
}
FIXTURE_JOURNAL_ACCOUNTS = [
    {"broker": "QUESTRADE", "account_number": "111", "account_label": "TFSA 111", "account_type": "TFSA",
     "tax_status": "TAX_FREE"},
    {"broker": "QUESTRADE", "account_number": "222", "account_label": "Margin 222", "account_type": "Margin",
     "tax_status": ""},
]
FIXTURE_TRADES = [
    {"trade_id": "T1", "account_number": "222", "account_label": "Margin 222", "symbol": "AMD", "direction": "SHORT",
     "quantity_opened": 100, "quantity_closed": 0, "average_entry_price": 150.0, "planned_stop": 152.5,
     "opened_at": "2026-09-28T07:00:00-07:00"},
    {"trade_id": "T2", "account_number": "111", "account_label": "TFSA 111", "symbol": "NVDA", "direction": "LONG",
     "quantity_opened": 50, "quantity_closed": 0, "average_entry_price": 120.0, "planned_stop": None,
     "opened_at": "2026-09-28T08:00:00-07:00"},
]


def fixture_sources(*, snapshot: Mapping[str, Any] | None = None, status: Mapping[str, Any] | None = None,
                    trades: list[Mapping[str, Any]] | None = None, max_positions: Any = None,
                    accounts: list[Mapping[str, Any]] | None = None) -> Sources:
    return Sources(
        snapshot=lambda: snapshot, status=lambda: status,
        open_trades=lambda: list(FIXTURE_TRADES if trades is None else trades),
        accounts=lambda: list(FIXTURE_JOURNAL_ACCOUNTS if accounts is None else accounts),
        industry_map=lambda: FIXTURE_INDUSTRIES, max_positions=lambda: max_positions,
    )


def fixture() -> Pack:
    from questrade_positions import fixture_snapshot

    fetched = FIXTURE_NOW.astimezone(timezone.utc).isoformat(timespec="seconds")
    return build(now=FIXTURE_NOW, sources=fixture_sources(snapshot=fixture_snapshot(fetched).as_dict()))
