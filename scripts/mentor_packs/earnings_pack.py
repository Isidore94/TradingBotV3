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
                            "description": "Tickers, e.g. ['NVDA', 'TSLA']; past 40 the rest are listed by name only."},
                "book": {"type": "array", "items": {"type": "string"},
                         "description": "Open positions: always listed, whatever the count."},
                "liked": {"type": "array", "items": {"type": "string"},
                          "description": "Liked names among the symbols (watch names, not positions)."},
                "focus": {"type": "array", "items": {"type": "string"},
                          "description": "Focus names among the symbols (watch names, not positions)."},
                "side": {"type": "string", "description": "LONG or SHORT when the group has one side."},
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


#: P16: how a row names where its symbol came from; only ``book`` is a position.
ORIGIN_WORDS = {"book": "book", "liked": "liked", "focus": "Focus"}


def origin_label(origin: str, side: str = "") -> str:
    """``book short`` / ``Focus long`` / ``liked``: what the row is, so a watch name is never called a position."""
    word = ORIGIN_WORDS.get(origin, "")
    return " ".join(part for part in (word, side.lower() if word else "") if part)


def build(symbols: Any = (), book: Any = (), liked: Any = (), focus: Any = (), side: str = "", *,
          now: datetime | None = None, paths: pick_pack.PickPaths | None = None) -> Pack:
    """One row per name: every ``book`` name, then the others up to 40 in all; the rest listed by name.

    File reads: call it on a worker.
    """
    held = _symbols(book)
    names = held + [sym for sym in _symbols(symbols) if sym not in held]
    likes, watched = set(_symbols(liked)), set(_symbols(focus))
    side_word = pick_pack._side(side)
    labelled = bool(held or likes or watched)

    def origin(sym: str) -> str:
        if not labelled:
            return ""
        return "book" if sym in held else "liked" if sym in likes else "focus" if sym in watched else "named"

    def tag(sym: str) -> str:
        label = origin_label(origin(sym), side_word)
        return f"{sym} ({label})" if label else sym
    if not names:
        return make_pack(NAME, (), empty_text="earnings_pack needs tickers, e.g. ['NVDA', 'TSLA']")
    moment = pick_pack._now(now)
    today = moment.astimezone(pick_pack.ET).date()
    src = paths or pick_pack.live_paths()
    # The open book is never cut; the tail (liked, then Focus) fills what is left of the 40.
    shown = names[:max(MAX_NAMES, len(held))]
    rest = names[len(shown):]
    head = (f"Earnings for {len(shown)} name(s) as of market date {today.isoformat()}: own next report and nearest "
            f"industry peer within {WINDOW_DAYS} days" + (f"; {len(rest)} more name(s) not listed" if rest else "")
            + ("; only rows marked 'book' are open positions, 'Focus' and 'liked' rows are watch names, never "
               "positions" if labelled else ""))
    rows: list[dict[str, Any]] = [{"id": "earn:asof", "kind": "asof", "text": head}]
    if rest:
        rows.append({"id": "earn:more", "kind": "more", "symbols": rest,
                     "text": f"{len(rest)} more not listed (no earnings read for them; say so, never guess): "
                             + ", ".join(rest)})
    try:
        dates = pick_pack._earnings_dates(src, today)
    except Exception as exc:  # noqa: BLE001 - an unreadable calendar is unknown for every name, never "none"
        why = type(exc).__name__
        rows += [{"id": f"earn:{sym}", "kind": "earnings", "symbol": sym, "origin": origin(sym),
                  "text": f"{tag(sym)}: earnings unknown (calendar unreadable: {why})"} for sym in shown]
        return make_pack(NAME, rows)
    try:
        loader = src.industry_map
        if loader is None:
            from industry_context import load_industry_context_map as loader
        industries = loader() or {}
    except Exception:  # noqa: BLE001 - peers are a label; unreadable = unknown per name
        industries = {}
    for sym in shown:
        rows.append({"id": f"earn:{sym}", "kind": "earnings", "symbol": sym, "origin": origin(sym),
                     "text": f"{tag(sym)}: {_own_text(sym, today, dates)}; {_peer_text(sym, today, dates, industries)}"})
    return make_pack(NAME, rows)


def fixture() -> Pack:
    with tempfile.TemporaryDirectory() as tmp:
        return build(["NVDA", "TSLA", "ZZZ"], now=pick_pack.FIXTURE_NOW, paths=pick_pack.write_fixture_world(tmp))
