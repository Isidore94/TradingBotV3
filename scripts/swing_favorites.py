"""The trader's hand-vetted swing picks - an append-only evidence store.

Trader, 2026-08-31: *"at the end of the day I have a list of my top swing
targets. I want a place to put them in so the bot knows my personal favourite
picks. They will usually become focus picks too but these ones get special
standing because I picked them by hand."*

That is a different act from the Master AVWAP like/dislike capture, which
records a verdict on a row the bot proposed. This records a name the trader
brought in themselves, so it is stored on its own terms:

* **Append-only, never rewritten.** An add is one row; a removal is a
  RETRACTION row that follows it. The file is the record of what the trader
  did, in the order they did it, so "added AMD then thought better of it" and
  "never added AMD" stay different facts.
* **Every row carries its own time twice** - ``event_at`` is tz-aware (machine
  order) and ``session_date`` is the market-local session (trading order). Same
  convention as ``evidence_ledger``, for the same reason: neither alone can
  answer "which session was this?" across an evening write. The packet that
  authorized this called the add timestamp ``added_at``; it is ``event_at``
  here because a retraction row carries one too and it is not an add.
* **The list is derived, not stored.** :func:`favorites_for_session` replays a
  session's rows in file order; the last action on a (symbol, side) wins.

Nothing in this module reads or writes Focus, a watchlist, a detector, a score
or ``review_policy.json``. The write-through to the swing Focus category is the
caller's second, separate write (``ui.services.swing_favorites_service``), and
it is deliberately the one that must not fail: a lost evidence row costs the
record, never the pick (ground rule: an evidence store never costs the thing it
records).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from project_paths import SWING_FAVORITES_FILE

#: Schema by NAME, never by number - a changed meaning is a new name.
SCHEMA_SWING_FAVORITE = "swing_favorite_v1"

ACTION_ADD = "add"
ACTION_REMOVE = "remove"
ACTIONS = (ACTION_ADD, ACTION_REMOVE)

#: The only origin this store writes. It exists so a later importer (a phone
#: note, a broker screen) is distinguishable from what the trader typed here.
ORIGIN_TRADER = "trader"

#: How far back the journal is asked about, in calendar days, when marking
#: which favorites were actually taken. Bounded on purpose: the widget shows
#: one session, and an unbounded query grows without limit against a store that
#: already holds a year of fills.
TAKEN_LOOKBACK_DAYS = 10


def normalize_side(side: object) -> str:
    """Map any side spelling to 'long' or 'short'; '' when it is neither."""
    text = str(side or "").strip().lower()
    if text.startswith("long") or text in {"l", "buy"}:
        return "long"
    if text.startswith("short") or text in {"s", "sell"}:
        return "short"
    return ""


def normalize_symbol(symbol: object) -> str:
    return str(symbol or "").strip().upper()


def parse_symbols(text: object) -> list[str]:
    """Symbols from typed or pasted text, order-preserving and de-duped.

    Reuses the watchlist parser so a paste out of TC2000 behaves here exactly
    as it does everywhere else in the desk.
    """
    from watchlist_utils import extract_watchlist_symbols

    seen: dict[str, None] = {}
    for symbol in extract_watchlist_symbols(str(text or "")):
        seen.setdefault(normalize_symbol(symbol), None)
    seen.pop("", None)
    return list(seen)


def current_session_date(now: datetime | None = None) -> str:
    """Today's market-local session date, as ``YYYY-MM-DD``."""
    try:
        from market_session import get_market_session_window

        return get_market_session_window(now).market_date.isoformat()
    except Exception:
        return (now or datetime.now()).date().isoformat()


def _event_time(now: datetime | None = None) -> str:
    moment = now or datetime.now()
    if moment.tzinfo is None:
        # astimezone() on a naive datetime attaches the machine's zone; it
        # never strips one. A row without an offset cannot be ordered against
        # one written on the other side of a DST change.
        moment = moment.astimezone()
    return moment.isoformat(timespec="seconds")


def build_row(
    *,
    symbol: object,
    side: object,
    action: str = ACTION_ADD,
    session_date: str = "",
    origin: str = ORIGIN_TRADER,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """One store row, or None when the symbol or side is unusable."""
    sym = normalize_symbol(symbol)
    side_text = normalize_side(side)
    action_text = str(action or "").strip().lower()
    if not sym or not side_text or action_text not in ACTIONS:
        return None
    return {
        "schema": SCHEMA_SWING_FAVORITE,
        "action": action_text,
        "session_date": str(session_date or "").strip() or current_session_date(now),
        "symbol": sym,
        "side": side_text,
        "event_at": _event_time(now),
        "origin": str(origin or ORIGIN_TRADER).strip() or ORIGIN_TRADER,
    }


def append_row(row: Mapping[str, Any], path: Path = SWING_FAVORITES_FILE) -> bool:
    """Append one row. Returns False when the write failed - never raises.

    A failed append loses the event, never the pick: the caller has already
    decided to place the name, and an unwritable evidence file must not turn
    that into an error the trader has to work around.
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    except OSError:
        return False
    return True


def record_favorite(
    *,
    symbol: object,
    side: object,
    action: str = ACTION_ADD,
    session_date: str = "",
    origin: str = ORIGIN_TRADER,
    now: datetime | None = None,
    path: Path = SWING_FAVORITES_FILE,
) -> dict[str, Any] | None:
    """Build and append one row. Returns the row, or None if nothing was written."""
    row = build_row(
        symbol=symbol,
        side=side,
        action=action,
        session_date=session_date,
        origin=origin,
        now=now,
    )
    if row is None:
        return None
    return row if append_row(row, path) else None


def load_rows(path: Path = SWING_FAVORITES_FILE) -> list[dict[str, Any]]:
    """Every row in file order (oldest first). A torn line is skipped, not raised."""
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def favorites_for_session(
    session_date: str = "",
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
    path: Path = SWING_FAVORITES_FILE,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """The live list for one session: adds replayed in file order, minus retractions.

    Order is the order the trader typed them in, because that is how they read
    the strip back - a list, not a set. A name added, removed and added again
    returns to the END of the list, which is where the trader just put it.
    """
    wanted = str(session_date or "").strip() or current_session_date(now)
    source = list(rows) if rows is not None else load_rows(path)
    live: dict[tuple[str, str], dict[str, Any]] = {}
    for row in source:
        if str(row.get("session_date") or "") != wanted:
            continue
        symbol = normalize_symbol(row.get("symbol"))
        side = normalize_side(row.get("side"))
        if not symbol or not side:
            continue
        key = (symbol, side)
        action = str(row.get("action") or "").strip().lower()
        if action == ACTION_REMOVE:
            live.pop(key, None)
        elif action == ACTION_ADD:
            live.pop(key, None)
            live[key] = {
                "symbol": symbol,
                "side": side,
                "session_date": wanted,
                "event_at": str(row.get("event_at") or ""),
                "origin": str(row.get("origin") or ORIGIN_TRADER),
            }
    return list(live.values())


# --------------------------------------------------------------- taken marks
# Read-only, display-only. It answers one question - "did I actually trade this
# one?" - and derives nothing else. No grading, no statistics, no per-tag
# anything: a number computed here would be a per-pick performance claim, and
# ground rule 10's statistics contract lives in `evidence_stats`, not in a chip.
def _trade_open_date(trade: Mapping[str, Any]) -> str:
    """The day a journal trade was opened, as ``YYYY-MM-DD``, or ''."""
    for field in ("opened_at", "trade_date"):
        text = str(trade.get(field) or "").strip()
        if len(text) >= 10:
            return text[:10]
    return ""


def taken_keys(
    favorites: Iterable[Mapping[str, Any]],
    trades: Iterable[Mapping[str, Any]],
) -> set[tuple[str, str]]:
    """Which (symbol, side) favorites have a journal trade opened on or after the pick.

    Matched on SYMBOL, not side: the journal records what was traded, and a
    favorite the trader took the other way round is still a favorite they
    acted on. A trade whose open date cannot be read marks nothing - an
    unmeasurable row is uncertainty, never confirmation.
    """
    opens: dict[str, str] = {}
    for trade in trades:
        symbol = normalize_symbol(trade.get("symbol"))
        opened = _trade_open_date(trade)
        if not symbol or not opened:
            continue
        # Keep the LATEST open per symbol; the test is ">= pick date", so the
        # latest is the one most likely to satisfy it.
        if opened > opens.get(symbol, ""):
            opens[symbol] = opened
    marks: set[tuple[str, str]] = set()
    for favorite in favorites:
        symbol = normalize_symbol(favorite.get("symbol"))
        side = normalize_side(favorite.get("side"))
        picked = str(favorite.get("session_date") or "").strip()
        if not symbol or not side or not picked:
            continue
        opened = opens.get(symbol, "")
        if opened and opened >= picked:
            marks.add((symbol, side))
    return marks


def taken_lookback_start(
    session_date: str = "",
    *,
    days: int = TAKEN_LOOKBACK_DAYS,
    now: datetime | None = None,
) -> date:
    """The earliest trade date the journal is asked about. Bounded on purpose."""
    from datetime import timedelta

    text = str(session_date or "").strip() or current_session_date(now)
    try:
        anchor = date.fromisoformat(text)
    except ValueError:
        anchor = (now or datetime.now()).date()
    return anchor - timedelta(days=max(0, int(days)))
