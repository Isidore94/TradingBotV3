"""Journal pack: the trader's trades for a day or a week, from the journal (``mode=ro``). Read-only.

``journal_pack(day="today"|"yesterday"|YYYY-MM-DD|weekday|"week"|"last_week"|"month"|"last_month")``. Per decision
(a stock trade, or option legs opened together = one spread): symbol, side, kind, open and
close times (ET), size, R on the planned risk or $ with "R unknown", hold time, and the
account's tax class. Then totals (count, wins, net R over the trades that have one, net $,
largest loss) and the open positions. Ids: ``jrn:<day>:<trade_id>`` (a spread joins its trade
ids with ``+``), ``jrn:<day>:totals``, ``jrn:<day>:open:<trade_id>``. A week's ``<day>`` is
``wk<monday>``; a month's is ``mo<YYYY-MM>``. A day's and a month's totals row comes first (a month's
carries the win rate), so the answer survives the attach budget. The journal store class is never built; missing data is "unknown".
"""

from __future__ import annotations

import re
import sqlite3
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from mentor_packs import journal_read
from mentor_packs.journal_read import ET
from mentor_packs.registry import Pack, make_pack

NAME = "journal_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "The trader's trades from his journal for one day, week or month: each trade's symbol, side, kind, open/close "
            "times (ET), size, R or $, hold time and account tax class; totals (count, wins, net R/$, largest "
            "loss) and open positions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "day": {"type": "string", "description": (
                    "'today' (default), 'yesterday', a date YYYY-MM-DD, a weekday name, 'week', 'last_week', "
                    "'month' or 'last_month'.")},
            },
            "required": [],
        },
    },
}

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
#: How far back ``recent_symbols`` looks for the trader's universe.
RECENT_SYMBOL_DAYS = 60


def live_journal() -> Path:
    from project_paths import JOURNAL_DB_FILE

    return Path(JOURNAL_DB_FILE)


def _now(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.astimezone()


def _previous_weekday(day: date) -> date:
    back = day - timedelta(days=1)
    while back.weekday() >= 5:
        back -= timedelta(days=1)
    return back


def resolve(day: Any, today: date) -> tuple[str, date, date] | None:
    """(label, first, last) market dates for the ``day`` argument; None when unreadable."""
    text = str(day or "today").strip().lower().replace(" ", "_")
    if text in ("", "today", "so_far", "this_morning"):
        return today.isoformat(), today, today
    if text == "yesterday":
        prior = _previous_weekday(today)
        return prior.isoformat(), prior, prior
    if text in ("week", "this_week"):
        monday = today - timedelta(days=today.weekday())
        return f"wk{monday.isoformat()}", monday, today
    if text == "last_week":
        monday = today - timedelta(days=today.weekday() + 7)
        return f"wk{monday.isoformat()}", monday, monday + timedelta(days=6)
    if text in ("month", "this_month"):
        first = today.replace(day=1)
        return f"mo{first:%Y-%m}", first, today
    if text == "last_month":
        last = today.replace(day=1) - timedelta(days=1)
        first = last.replace(day=1)
        return f"mo{first:%Y-%m}", first, last
    if text.rstrip("s") in _WEEKDAYS:
        target = _WEEKDAYS.index(text.rstrip("s"))
        back = today - timedelta(days=(today.weekday() - target) % 7)
        return back.isoformat(), back, back
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            one = date.fromisoformat(text)
        except ValueError:
            return None
        return one.isoformat(), one, one
    return None


def _in(moment: datetime | None, first: date, last: date) -> bool:
    return moment is not None and first <= moment.date() <= last


def _money(value: float | None) -> str:
    return "unknown" if value is None else f"{value:+,.2f}"


def _hold(opened: datetime | None, closed: datetime | None) -> str:
    if opened is None or closed is None:
        return "unknown"
    minutes = int((closed - opened).total_seconds() // 60)
    if minutes < 60:
        return f"{minutes} min"
    if minutes < 24 * 60:
        return f"{minutes // 60} h {minutes % 60} min"
    return f"{minutes // (24 * 60)} d"


def _accounts(path: Path) -> dict[str, str]:
    """``{account_number: tax class text}`` from the journal's accounts (none = {})."""
    from mentor_packs import book_pack

    out: dict[str, str] = {}
    for account in journal_read.read_accounts(path):
        number = str(account.get("account_number") or "")
        if number:
            out[number] = book_pack.CLASS_TEXT.get(book_pack.tax_class(account), "account type unknown")
    return out


def _qty(trades: Iterable[Mapping[str, Any]]) -> str:
    sizes = [journal_read.num(t.get("quantity_opened")) for t in trades]
    return "unknown" if any(s is None for s in sizes) else "/".join(f"{s:g}" for s in sizes)


def _unit_row(label: str, unit: journal_read.Unit, accounts: Mapping[str, str]) -> dict[str, Any]:
    first = unit.trades[0]
    side = str(first.get("direction") or "?").upper() if len(unit.trades) == 1 else "SPREAD"
    kind = "spread" if unit.kind == "option" and len(unit.trades) > 1 else unit.kind
    result = (f"{unit.r:+.2f}R ({_money(unit.pnl)} $)" if unit.r is not None
              else f"{_money(unit.pnl)} $ (R unknown: no planned stop)")
    opened = unit.opened.strftime("%a %H:%M") if unit.opened else "unknown"
    closed = unit.closed.strftime("%a %H:%M") if unit.closed else "unknown"
    account = accounts.get(unit.account_number, "account type unknown")
    return {
        "id": f"jrn:{label}:{'+'.join(unit.ids)}",
        "kind": "trade",
        "symbol": unit.symbol,
        "pnl": unit.pnl,
        "r": unit.r,
        "text": (f"{side} {unit.symbol} ({kind}) size {_qty(unit.trades)}, opened {opened} ET, closed {closed} ET, "
                 f"held {_hold(unit.opened, unit.closed)}: {result}; account {unit.account_number or '?'} ({account})"),
    }


def build(day: Any = "today", *, now: datetime | None = None, journal: Path | str | None = None) -> Pack:
    """Build the journal pack. Journal reads (``mode=ro``): call it on a worker."""
    moment = _now(now)
    today = moment.astimezone(ET).date()
    span = resolve(day, today)
    if span is None:
        return make_pack(NAME, (), empty_text=f"journal_pack could not read the day {day!r}; try 'today' or YYYY-MM-DD")
    label, first, last = span
    path = Path(journal) if journal is not None else live_journal()
    try:
        trades = journal_read.read_trades(path, since=first.isoformat())
        # Every open position, however old (the dated read above only sees trades touched since ``first``).
        from mentor_packs.gate_pack import read_open_trades

        open_trades = read_open_trades(path)
        accounts = _accounts(path)
    except sqlite3.Error as exc:
        return make_pack(NAME, (), empty_text=f"the journal could not be read ({type(exc).__name__}); unknown")
    when = label if first == last else f"{first.isoformat()} to {last.isoformat()}"
    rows: list[dict[str, Any]] = []
    closed_units = [
        unit for unit in journal_read.units(trades)
        if _in(unit.closed, first, last) or _in(unit.opened, first, last)
    ]
    for unit in closed_units:
        rows.append(_unit_row(label, unit, accounts))
    opened_in_span = [t for t in open_trades if _in(journal_read.parse_time(t.get("opened_at")), first, last)]
    values = [unit.pnl for unit in closed_units]
    known_values = [v for v in values if v is not None]
    rs = [unit.r for unit in closed_units if unit.r is not None]
    wins = sum(1 for v in known_values if v > 0)
    worst = min(closed_units, key=lambda u: u.pnl if u.pnl is not None else float("inf"), default=None)
    worst_text = "none"
    if worst is not None and worst.pnl is not None and worst.pnl < 0:
        worst_text = f"{worst.symbol} {_money(worst.pnl)} $" + (f" ({worst.r:+.2f}R)" if worst.r is not None else "")
    net_r = f"{sum(rs):+.2f}R over {len(rs)} of {len(closed_units)} with a planned stop" if rs else "R unknown"
    if not closed_units and not opened_in_span:
        totals = f"{when}: no trades in the journal (closed or opened); {len(open_trades)} open position(s) overall"
    else:
        totals = (
            f"{when}: {len(closed_units)} closed trade(s), {wins} win(s); net {_money(sum(known_values)) if known_values else 'unknown'} $"
            f" ({net_r}); largest loss {worst_text}; {len(opened_in_span)} opened and still open"
            + ("" if len(known_values) == len(values) else f"; {len(values) - len(known_values)} with no PnL (unknown)")
        )
    month = label.startswith("mo")
    if month and known_values:
        totals += f"; win rate {100 * wins / len(known_values):.0f}% ({wins} of {len(known_values)} with a PnL)"
    total_row = {"id": f"jrn:{label}:totals", "kind": "totals", "count": len(closed_units), "wins": wins,
                 "net": sum(known_values) if known_values else None, "open": len(open_trades), "text": totals}
    # A day or a month answers from its totals first (a wrong premise shows at once; a tight budget keeps it).
    rows.insert(len(rows) if label.startswith("wk") else 0, total_row)
    for trade in open_trades:
        opened = journal_read.parse_time(trade.get("opened_at"))
        qty = (journal_read.num(trade.get("quantity_opened")) or 0) - (journal_read.num(trade.get("quantity_closed")) or 0)
        entry = journal_read.num(trade.get("average_entry_price"))
        stop = journal_read.num(trade.get("planned_stop"))
        account = accounts.get(str(trade.get("account_number") or ""), "account type unknown")
        rows.append({
            "id": f"jrn:{label}:open:{trade.get('trade_id')}",
            "kind": "open",
            "symbol": str(trade.get("symbol") or "").upper(),
            "text": (f"Open {str(trade.get('direction') or '?').upper()} {trade.get('symbol')} {qty:g} @ "
                     f"{'unknown' if entry is None else f'{entry:.2f}'}, stop {'none' if stop is None else f'{stop:.2f}'}"
                     f", since {opened.strftime('%a %Y-%m-%d %H:%M') + ' ET' if opened else 'unknown'}; "
                     f"account {trade.get('account_number') or '?'} ({account})"),
        })
    return make_pack(NAME, rows)


def today_summary(journal: Path | str | None = None, *, now: datetime | None = None) -> str:
    """One line for the context pack: "today so far: N trades, net ..., k open"."""
    pack = build("today", now=now, journal=journal)
    totals = next((row for row in pack.rows if row.get("kind") == "totals"), None)
    if totals is None:
        return f"Today so far: unknown ({pack.empty_text or 'journal not read'})"
    net = totals.get("net")
    return (f"Today so far: {totals['count']} closed trade(s), {totals['wins']} win(s), net "
            f"{_money(net) + ' $' if net is not None else 'unknown'}, {totals['open']} open")


def recent_symbols(journal: Path | str | None = None, *, now: datetime | None = None,
                   days: int = RECENT_SYMBOL_DAYS) -> list[str]:
    """Stock symbols (option underlyings) the trader traded in the last ``days`` days."""
    path = Path(journal) if journal is not None else live_journal()
    since = (_now(now).astimezone(ET).date() - timedelta(days=days)).isoformat()
    found: list[str] = []
    for trade in journal_read.read_trades(path, since=since):
        sym = journal_read.underlying(trade.get("symbol")) if journal_read.is_option(trade) else str(
            trade.get("symbol") or "").strip().upper()
        if sym and sym not in found:
            found.append(sym)
    return found


# ---------------------------------------------------------------- fixture
FIXTURE_NOW = datetime(2026, 9, 30, 11, 0, tzinfo=ET)  # Wed, 08:00 PT


def write_fixture_journal(path: Path | str) -> Path:
    """Wed 2026-09-30: an NVDA long won 1R, an AMD short lost $120 (no stop), a two-leg ALL put spread in a TFSA,
    an MSFT long still open. Tue 2026-09-29: one TSLA short lost 0.5R."""
    target = Path(path)
    conn = sqlite3.connect(target)
    conn.executescript(
        "CREATE TABLE trades (trade_id TEXT, account_number TEXT, account_label TEXT, symbol TEXT, security_type TEXT,"
        " direction TEXT, status TEXT, opened_at TEXT, closed_at TEXT, quantity_opened REAL, quantity_closed REAL,"
        " average_entry_price REAL, average_exit_price REAL, net_pnl REAL, net_pnl_usd REAL);"
        "CREATE TABLE trade_annotations (trade_id TEXT, planned_stop REAL, planned_risk REAL);"
        "CREATE TABLE accounts (broker TEXT, account_number TEXT, account_label TEXT, account_type TEXT,"
        " tax_status TEXT);"
    )
    rows = [
        ("W1", "M1", "Margin", "NVDA", "STK", "LONG", "CLOSED", "2026-09-30T09:35:00-04:00",
         "2026-09-30T10:20:00-04:00", 100, 100, 120.0, 121.0, 100.0, 100.0),
        ("W2", "M1", "Margin", "AMD", "STK", "SHORT", "CLOSED", "2026-09-30T09:50:00-04:00",
         "2026-09-30T10:05:00-04:00", 50, 50, 150.0, 152.4, -120.0, -120.0),
        ("W3", "T9", "TFSA", "ALL261016P00180000", "OPT", "LONG", "CLOSED", "2026-09-30T10:00:00-04:00",
         "2026-09-30T10:40:00-04:00", 1, 1, 3.0, 4.0, 100.0, 100.0),
        ("W4", "T9", "TFSA", "ALL261016P00170000", "OPT", "SHORT", "CLOSED", "2026-09-30T10:01:00-04:00",
         "2026-09-30T10:40:00-04:00", 1, 1, 1.5, 2.0, -50.0, -50.0),
        ("W5", "M1", "Margin", "MSFT", "STK", "LONG", "OPEN", "2026-09-30T10:30:00-04:00", "", 10, 0, 400.0, 0,
         0, 0),
        ("Y1", "M1", "Margin", "TSLA", "STK", "SHORT", "CLOSED", "2026-09-29T09:40:00-04:00",
         "2026-09-29T11:00:00-04:00", 20, 20, 250.0, 251.0, -20.0, -20.0),
    ]
    conn.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.executemany("INSERT INTO trade_annotations VALUES (?,?,?)",
                     [("W1", 119.0, None), ("Y1", 252.0, None), ("W5", 395.0, None)])
    conn.executemany("INSERT INTO accounts VALUES (?,?,?,?,?)",
                     [("questrade", "M1", "Margin", "Margin", "TAXABLE"), ("questrade", "T9", "TFSA", "TFSA", "")])
    conn.commit()
    conn.close()
    return target


def fixture() -> Pack:
    with tempfile.TemporaryDirectory() as tmp:
        return build("today", now=FIXTURE_NOW, journal=write_fixture_journal(Path(tmp) / "trade_journal.sqlite3"))
