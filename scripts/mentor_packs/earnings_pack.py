"""Earnings pack (P14): own and nearest-peer earnings for many names, one row per name. Read-only.

``earnings_pack(symbols)`` answers "anything reporting in my longs / book?" over the whole
list (the app passes the open book, Focus and liked names, at most 40) so the model never
guesses about a name it did not see. Built from the same earnings calendar and industry map
as ``pick_pack``. Ids: ``earn:asof`` and ``earn:<SYM>``. A name with no date in the calendar
says "unknown", never "no earnings".
"""

from __future__ import annotations

import re
import tempfile
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping

from mentor_packs import pick_pack
from mentor_packs.registry import Pack, make_pack

NAME = "earnings_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Earnings for many names at once, one line each: the name's own next report and its nearest "
            "industry peer reporting within 14 days. Use it for 'anything reporting in my longs/book?'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbols": {"type": "array", "items": {"type": "string"},
                            "description": "Tickers, e.g. ['NVDA', 'TSLA'] (at most 40)."},
            },
            "required": ["symbols"],
        },
    },
}

WINDOW_DAYS = 14
MAX_NAMES = 40


def _symbols(symbols: Any) -> list[str]:
    """A list, or a comma/space separated string, to unique uppercase tickers in order."""
    items: Iterable[Any] = re.split(r"[\s,;]+", symbols) if isinstance(symbols, str) else (symbols or ())
    out: list[str] = []
    for item in items:
        sym = pick_pack._sym(item).lstrip("$")
        if sym and sym.replace(".", "").replace("-", "").isalnum() and sym not in out:
            out.append(sym)
    return out


def _own_text(sym: str, today: date, dates: Mapping[str, list[date]]) -> str:
    upcoming = [value for value in dates.get(sym, ()) if value >= today]
    if not upcoming:
        return "own report unknown (no date in the calendar)"
    nxt = upcoming[0]
    days = (nxt - today).days
    if days <= WINDOW_DAYS:
        return f"REPORTS {nxt:%a %Y-%m-%d}, {pick_pack._day_word(days)}"
    return f"no own report within {WINDOW_DAYS} days (next {nxt:%Y-%m-%d})"


def _peer_text(sym: str, today: date, dates: Mapping[str, list[date]], industries: Mapping[str, Any]) -> str:
    context = industries.get(sym) or {}
    members = {pick_pack._sym(p) for p in context.get("industry_member_symbols") or ()} - {sym, ""}
    if not members:
        return "peers unknown (no industry)"
    hi = today + timedelta(days=WINDOW_DAYS)
    near = sorted(((value, peer) for peer in members for value in dates.get(peer, ()) if today <= value <= hi))
    if not near:
        return f"no peer reports within {WINDOW_DAYS} days"
    when, peer = near[0]
    return f"nearest peer {peer} {when:%a %Y-%m-%d}, {pick_pack._day_word((when - today).days)}"


def build(symbols: Any = (), *, now: datetime | None = None, paths: pick_pack.PickPaths | None = None) -> Pack:
    """One row per name (at most 40). File reads: call it on a worker."""
    names = _symbols(symbols)
    if not names:
        return make_pack(NAME, (), empty_text="earnings_pack needs tickers, e.g. ['NVDA', 'TSLA']")
    moment = pick_pack._now(now)
    today = moment.astimezone(pick_pack.ET).date()
    src = paths or pick_pack.live_paths()
    shown, extra = names[:MAX_NAMES], len(names) - MAX_NAMES
    head = (f"Earnings for {len(shown)} name(s) as of market date {today.isoformat()}: own next report and nearest "
            f"industry peer within {WINDOW_DAYS} days" + (f"; {extra} more name(s) not listed" if extra > 0 else ""))
    rows: list[dict[str, Any]] = [{"id": "earn:asof", "kind": "asof", "text": head}]
    try:
        dates = pick_pack._earnings_dates(src, today)
    except Exception as exc:  # noqa: BLE001 - an unreadable calendar is unknown for every name, never "none"
        why = type(exc).__name__
        rows += [{"id": f"earn:{sym}", "kind": "earnings", "symbol": sym,
                  "text": f"{sym}: earnings unknown (calendar unreadable: {why})"} for sym in shown]
        return make_pack(NAME, rows)
    try:
        loader = src.industry_map
        if loader is None:
            from industry_context import load_industry_context_map as loader
        industries = loader() or {}
    except Exception:  # noqa: BLE001 - peers are a label; unreadable = unknown per name
        industries = {}
    for sym in shown:
        rows.append({"id": f"earn:{sym}", "kind": "earnings", "symbol": sym,
                     "text": f"{sym}: {_own_text(sym, today, dates)}; {_peer_text(sym, today, dates, industries)}"})
    return make_pack(NAME, rows)


def fixture() -> Pack:
    with tempfile.TemporaryDirectory() as tmp:
        return build(["NVDA", "TSLA", "ZZZ"], now=pick_pack.FIXTURE_NOW, paths=pick_pack.write_fixture_world(tmp))
