"""Tilt pack: today's trading patterns after a loss, as observations with leg ids. Read-only.

Reads today's ``trade_legs`` and ``trades`` (``mode=ro``) and emits, with the leg ids as
evidence:

- ``tilt:burst:<HHMMSS>``: BURST_OPENS or more opens within BURST_WINDOW after a losing close;
- ``tilt:reentry:<SYM>:<HHMMSS>``: the same symbol and side re-opened within REENTRY_WINDOW of a loss on it;
- ``tilt:size:<SYM>:<HHMMSS>``: an open at SIZE_MULTIPLE x the day's median open size (so far) after a loss;
- ``tilt:streak:<HHMMSS>``: STREAK_LOSSES losing closes in a row today;
- ``tilt:base:<kind>``: over the last BASE_SESSIONS sessions with trades, how often the pattern
  was followed by a red rest of day (n, Wilson LB, "too few" under the floor).

A losing close is a trade closed today with a known net PnL below zero; a trade still open
has no PnL yet (unknown, never a loss). Observations only: never a rule, never an order.
The trader can overrule every threshold. Call :func:`build` on a worker.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from mentor_packs import journal_read
from mentor_packs.journal_read import ET
from mentor_packs.registry import Pack, make_pack

NAME = "tilt_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Today's in-session patterns after a loss (a burst of opens, a re-entry on the same name, a "
            "bigger size, a losing streak), each with the journal leg ids, plus how often each pattern was "
            "followed by a red rest of day over the last 60 sessions (n; 'too few' under 30). Observations only."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

#: Three entries soon after a loss is more than one planned re-entry.
BURST_OPENS = 3
#: Ten minutes is shorter than any setup's own confirmation time on M5.
BURST_WINDOW = timedelta(minutes=10)
#: Fifteen minutes = three M5 bars: a re-entry faster than that did not wait for a new setup.
REENTRY_WINDOW = timedelta(minutes=15)
#: 1.5x the day's median open size is a clear step up, not fill noise.
SIZE_MULTIPLE = 1.5
#: Three losing closes in a row is the first streak the Trader Mirror could see at all.
STREAK_LOSSES = 3
#: About three months of sessions: enough days for a base rate, recent enough to be the same trader.
BASE_SESSIONS = 60
#: Partial fills of one order land as separate legs a few seconds apart: one open.
FILL_MERGE = timedelta(seconds=60)
KINDS = ("burst", "reentry", "size", "streak")
KIND_TEXT = {"burst": "a burst of opens after a loss", "reentry": "a re-entry on the same name after a loss",
             "size": "a bigger open after a loss", "streak": f"{STREAK_LOSSES} losing closes in a row"}
FOOTER = "Observation, not a rule."


def min_reportable_n() -> int:
    from evidence_stats import MIN_REPORTABLE_N

    return int(MIN_REPORTABLE_N)


# ---------------------------------------------------------------- one day's events
@dataclass
class Event:
    at: datetime
    trade_id: str
    symbol: str
    side: str
    legs: list[int]
    notional: float | None = None
    pnl: float | None = None


def _side(value: Any) -> str:
    text = str(value or "").strip().upper()
    return {"BUY": "LONG", "SELL": "SHORT"}.get(text, text)


def day_events(legs: list[Mapping[str, Any]], trades: list[Mapping[str, Any]], day: date,
               until: datetime | None = None) -> tuple[list[Event], list[Event]]:
    """(opens, closes) on ``day`` up to ``until``: opens merged per trade within FILL_MERGE; closes with PnL."""
    opens: list[Event] = []
    last: dict[str, Event] = {}
    close_legs: dict[str, list[int]] = {}
    for leg in legs:
        at = journal_read.parse_time(leg.get("timestamp"))
        if at is None or at.date() != day or (until is not None and at > until):
            continue
        trade_id = str(leg.get("trade_id") or "")
        role = str(leg.get("role") or "").upper()
        if role == "CLOSE":
            close_legs.setdefault(trade_id, []).append(int(leg["leg_id"]))
            continue
        if role != "OPEN":
            continue
        qty, price = journal_read.num(leg.get("quantity")), journal_read.num(leg.get("price"))
        mult = journal_read.OPTION_MULTIPLIER if journal_read.is_option(leg) else 1.0
        notional = abs(qty * price * mult) if qty is not None and price is not None else None
        prior = last.get(trade_id)
        if prior is not None and at - prior.at <= FILL_MERGE:
            prior.legs.append(int(leg["leg_id"]))
            prior.notional = None if prior.notional is None or notional is None else prior.notional + notional
            continue
        event = Event(at, trade_id, str(leg.get("symbol") or "").upper(), _side(leg.get("direction")),
                      [int(leg["leg_id"])], notional)
        opens.append(event)
        last[trade_id] = event
    closes: list[Event] = []
    for trade in trades:
        at = journal_read.parse_time(trade.get("closed_at"))
        value = journal_read.pnl(trade)
        if (str(trade.get("status") or "").upper() != "CLOSED" or at is None or at.date() != day or value is None
                or (until is not None and at > until)):
            continue
        trade_id = str(trade.get("trade_id") or "")
        closes.append(Event(at, trade_id, str(trade.get("symbol") or "").upper(), _side(trade.get("direction")),
                            sorted(close_legs.get(trade_id, [])), pnl=value))
    opens.sort(key=lambda e: (e.at, e.legs))
    closes.sort(key=lambda e: (e.at, e.trade_id))
    return opens, closes


def _stamp(moment: datetime) -> str:
    return f"{moment.astimezone(ET):%H%M%S}"


def _legs_text(legs: list[int]) -> str:
    return "legs " + ", ".join(str(leg) for leg in legs) if legs else "no leg ids"


def _minutes(delta: timedelta) -> int:
    return max(1, int(-(-delta.total_seconds() // 60)))


def detect(opens: list[Event], closes: list[Event]) -> list[dict[str, Any]]:
    """Every pattern in one day's events, oldest first. Deterministic: same events, same rows."""
    found: list[dict[str, Any]] = []
    losses = [c for c in closes if c.pnl is not None and c.pnl < 0]
    for loss in losses:
        after = [o for o in opens if loss.at < o.at <= loss.at + BURST_WINDOW]
        if len(after) >= BURST_OPENS:
            legs = loss.legs + [leg for o in after for leg in o.legs]
            found.append({"kind": "burst", "id": f"tilt:burst:{_stamp(loss.at)}", "at": after[BURST_OPENS - 1].at,
                          "symbol": "", "legs": legs,
                          "text": (f"{len(after)} opens in {_minutes(after[-1].at - loss.at)} min after a losing "
                                   f"close on {loss.symbol} ({_legs_text(legs)}). {FOOTER}")})
        again = next((o for o in opens if loss.at < o.at <= loss.at + REENTRY_WINDOW
                      and o.symbol == loss.symbol and o.side == loss.side), None)
        if again is not None:
            legs = loss.legs + again.legs
            found.append({"kind": "reentry", "id": f"tilt:reentry:{again.symbol}:{_stamp(again.at)}", "at": again.at,
                          "symbol": again.symbol, "legs": legs,
                          "text": (f"Re-opened {again.symbol} {again.side} {_minutes(again.at - loss.at)} min after "
                                   f"a losing close on it ({_legs_text(legs)}). {FOOTER}")})
    first_loss = losses[0].at if losses else None
    for index, event in enumerate(opens):
        sizes = [o.notional for o in opens[: index + 1] if o.notional is not None]
        if first_loss is None or event.at <= first_loss or event.notional is None or len(sizes) < 2:
            continue
        middle = median(sizes)
        if middle > 0 and event.notional >= SIZE_MULTIPLE * middle:
            loss = [c for c in losses if c.at < event.at][-1]
            legs = loss.legs + event.legs
            found.append({"kind": "size", "id": f"tilt:size:{event.symbol}:{_stamp(event.at)}", "at": event.at,
                          "symbol": event.symbol, "legs": legs,
                          "text": (f"An open of {event.symbol} at {event.notional / middle:.1f}x today's median size "
                                   f"after a loss ({_legs_text(legs)}). {FOOTER}")})
    run: list[Event] = []
    for close in closes:
        run = run + [close] if close.pnl is not None and close.pnl < 0 else []
        if len(run) == STREAK_LOSSES:
            legs = [leg for c in run for leg in c.legs]
            found.append({"kind": "streak", "id": f"tilt:streak:{_stamp(run[0].at)}", "at": close.at, "symbol": "",
                          "legs": legs, "text": f"{STREAK_LOSSES} losing closes in a row ({_legs_text(legs)}). {FOOTER}"})
    seen: dict[str, int] = {}
    for row in found:  # two patterns in one second get distinct ids
        seen[row["id"]] = seen.get(row["id"], 0) + 1
        if seen[row["id"]] > 1:
            row["id"] = f"{row['id']}-{seen[row['id']]}"
    found.sort(key=lambda r: (r["at"], r["id"]))
    return found


def realized(closes: list[Event], *, before: datetime | None = None, after: datetime | None = None) -> float:
    return float(sum(c.pnl or 0.0 for c in closes
                     if (before is None or c.at <= before) and (after is None or c.at > after)))


# ---------------------------------------------------------------- base rates (history)
_base_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
_base_lock = threading.Lock()


def _wilson(wins: int, n: int) -> float | None:
    import setup_grades

    return setup_grades.wilson_lower_bound(wins, n)


def base_rows(legs: list[Mapping[str, Any]], trades: list[Mapping[str, Any]], today: date,
              floor: int) -> list[dict[str, Any]]:
    """Per kind: of the times it happened in the last BASE_SESSIONS trading days, how many had a red rest of day."""
    days = sorted({d for leg in legs if (t := journal_read.parse_time(leg.get("timestamp"))) is not None
                   and (d := t.date()) < today})[-BASE_SESSIONS:]
    counts = {kind: [0, 0] for kind in KINDS}  # [n, red]
    for day in days:
        opens, closes = day_events(legs, trades, day)
        for row in detect(opens, closes):
            counts[row["kind"]][0] += 1
            counts[row["kind"]][1] += 1 if realized(closes, after=row["at"]) < 0 else 0
    rows = []
    for kind in KINDS:
        n, red = counts[kind]
        head = (f"Base rate over the last {len(days)} session(s) with trades: {KIND_TEXT[kind]} was followed by "
                f"a red rest of day ")
        if n < floor:
            text = head + f"{red} of {n} times: too few (n={n}, floor {floor})"
        else:
            lb = _wilson(red, n)
            text = head + f"{red / n:.0%} of the time (n={n}, LB {lb:.2f})"
        rows.append({"id": f"tilt:base:{kind}", "kind": "base", "pattern": kind, "n": n, "red": red,
                     "sessions": len(days), "too_few": n < floor, "text": text})
    return rows


# ---------------------------------------------------------------- build
def _now(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.astimezone()


def live_journal() -> Path:
    from project_paths import JOURNAL_DB_FILE

    return Path(JOURNAL_DB_FILE)


def today_observations(journal: Path | str, now: datetime) -> tuple[list[dict[str, Any]], list[Event], list[Event]]:
    """(observations, opens, closes) for today (New York) up to ``now``."""
    day = now.astimezone(ET).date()
    since = day.isoformat()
    legs = journal_read.read_legs(journal, since=since)
    trades = journal_read.read_trades(journal, since=since)
    opens, closes = day_events(legs, trades, day, until=now)
    rows = detect(opens, closes)
    for row in rows:
        row["before_pnl"] = round(realized(closes, before=row["at"]), 2)
    return rows, opens, closes


def build(*, now: datetime | None = None, journal: Path | str | None = None) -> Pack:
    """Build the tilt pack. Journal reads (``mode=ro``): call it on a worker."""
    moment = _now(now)
    path = Path(journal) if journal is not None else live_journal()
    day = moment.astimezone(ET).date()
    floor = min_reportable_n()
    try:
        observed, opens, closes = today_observations(path, moment)
    except sqlite3.Error as exc:
        return make_pack(NAME, (), empty_text=f"the journal could not be read ({type(exc).__name__}); unknown")
    rows: list[dict[str, Any]] = [{
        "id": "tilt:asof", "kind": "asof", "asof_utc": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "text": (f"Today {day.isoformat()} up to {moment.astimezone(ET):%H:%M} ET: {len(opens)} open(s), "
                 f"{len(closes)} close(s) with a PnL; realized {realized(closes):+.2f}"),
    }]
    for row in observed:
        rows.append({**row, "at": row["at"].isoformat(timespec="seconds")})
    if not observed:
        rows.append({"id": "tilt:none", "kind": "none", "text": "No pattern after a loss today"})
    key = (str(path), day.isoformat())
    with _base_lock:
        base = _base_cache.get(key)
    if base is None:
        since = (day - timedelta(days=BASE_SESSIONS * 7 // 5 + 14)).isoformat()
        base = base_rows(journal_read.read_legs(path, since=since), journal_read.read_trades(path, since=since), day,
                         floor)
        with _base_lock:
            _base_cache[key] = base
    rows += base
    return make_pack(NAME, rows)


def observations(pack: Pack) -> list[dict[str, Any]]:
    return [row for row in pack.rows if str(row.get("kind")) in KINDS]


# ---------------------------------------------------------------- fixture
FIXTURE_NOW = datetime(2026, 9, 29, 10, 30, tzinfo=ET)  # Tue, 07:30 PT


def _fixture_day(day: date, start_leg: int, plan: list[tuple]) -> tuple[list[tuple], list[tuple]]:
    """``plan`` rows: (trade_id, symbol, side, qty, price, open HH:MM[:SS], close HH:MM or '', pnl, extra fills)."""
    trades, legs = [], []
    leg = start_leg
    for trade_id, symbol, side, qty, price, opened, closed, pnl, fills in plan:
        at = datetime.combine(day, datetime.strptime(opened, "%H:%M:%S" if opened.count(":") == 2 else "%H:%M").time(),
                              tzinfo=ET)
        legs.append((leg, trade_id, "BUY" if side == "LONG" else "SELL", "OPEN", qty, price, at.isoformat()))
        leg += 1
        for seconds in fills:
            legs.append((leg, trade_id, "BUY" if side == "LONG" else "SELL", "OPEN", qty, price,
                         (at + timedelta(seconds=seconds)).isoformat()))
            leg += 1
        close_at = datetime.combine(day, datetime.strptime(closed, "%H:%M").time(), tzinfo=ET) if closed else None
        if close_at is not None:
            legs.append((leg, trade_id, "SELL" if side == "LONG" else "BUY", "CLOSE", qty, price, close_at.isoformat()))
            leg += 1
        trades.append((trade_id, "M1", symbol, "STK", side, "CLOSED" if closed else "OPEN", at.isoformat(),
                       close_at.isoformat() if close_at else "", qty, pnl, pnl))
    return trades, legs


def write_fixture_journal(path: Path | str) -> Path:
    """Today (Tue 2026-09-29): a loss on AMD at 09:40, then AMD re-opened at 4x size, NVDA and TSLA within
    9 min (a burst), and three losing closes in a row. Two earlier days carry one streak each."""
    target = Path(path)
    conn = sqlite3.connect(target)
    conn.executescript(
        "CREATE TABLE trades (trade_id TEXT, account_number TEXT, symbol TEXT, security_type TEXT, direction TEXT, "
        "status TEXT, opened_at TEXT, closed_at TEXT, quantity_opened REAL, net_pnl REAL, net_pnl_usd REAL);"
        "CREATE TABLE trade_legs (leg_id INTEGER PRIMARY KEY, trade_id TEXT, side TEXT, role TEXT, quantity REAL, "
        "price REAL, timestamp TEXT);"
    )
    today = FIXTURE_NOW.date()
    days = [
        (date(2026, 9, 25), 100, [("H1", "AAA", "LONG", 10, 10.0, "09:35", "09:40", -5.0, ()),
                                  ("H2", "BBB", "LONG", 10, 10.0, "09:41", "09:50", -5.0, ()),
                                  ("H3", "CCC", "LONG", 10, 10.0, "09:51", "10:00", -5.0, ()),
                                  ("H4", "DDD", "LONG", 10, 10.0, "10:01", "11:00", -8.0, ())]),
        (date(2026, 9, 28), 200, [("J1", "AAA", "SHORT", 10, 10.0, "09:35", "09:40", -5.0, ()),
                                  ("J2", "BBB", "SHORT", 10, 10.0, "09:41", "09:50", -5.0, ()),
                                  ("J3", "CCC", "SHORT", 10, 10.0, "09:51", "10:00", -5.0, ()),
                                  ("J4", "DDD", "SHORT", 10, 10.0, "10:01", "11:00", 30.0, ())]),
        (today, 1, [("T1", "AMD", "LONG", 100, 150.0, "09:31", "09:40", -120.0, ()),
                    ("T2", "AMD", "LONG", 400, 150.0, "09:45", "", None, ()),
                    ("T3", "NVDA", "LONG", 10, 120.0, "09:46", "10:00", -30.0, (20,)),
                    ("T4", "TSLA", "SHORT", 5, 250.0, "09:49", "10:05", -20.0, ()),
                    ("T5", "MSFT", "LONG", 10, 400.0, "10:10", "10:20", 50.0, ())]),
    ]
    for day, start, plan in days:
        trades, legs = _fixture_day(day, start, plan)
        conn.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?)", trades)
        conn.executemany("INSERT INTO trade_legs VALUES (?,?,?,?,?,?,?)", legs)
    conn.commit()
    conn.close()
    return target


def fixture() -> Pack:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = write_fixture_journal(Path(tmp) / "trade_journal.sqlite3")
        pack = build(now=FIXTURE_NOW, journal=path)
        with _base_lock:
            _base_cache.pop((str(path), FIXTURE_NOW.date().isoformat()), None)
        return pack
