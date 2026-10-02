"""RS pack: where the relative strength is, from the desk's industry and sector board files. Read-only.

Reads only the board snapshot (``INDUSTRY_BOARD_STATE_FILE``), ``industry_indexes.csv``,
``sector_indexes.csv`` and the symbol -> industry map (``industry_context``). Each leading /
lagging row says which of the trader's names sit in that group (book, liked, Focus). A missing
board, or one older than one session, is an ``rs:none`` row with its age, never a guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.registry import Pack, make_pack

NAME = "rs_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Relative strength from the desk's industry board: the leading and lagging industries (or sectors) "
            "with rank, 1d and 5d change, member count and which of the trader's names (book, liked, Focus) sit "
            "in each, plus the rank of every industry his open book touches."
        ),
        "parameters": {"type": "object", "properties": {
            "level": {"type": "string", "enum": ["industry", "sector"], "description": "industry (default) or sector"},
            "top": {"type": "integer", "description": "how many leaders and laggards (default 5)"},
        }, "required": []},
    },
}

ET = ZoneInfo("America/New_York")
#: The board's ``last_success_at`` is written naive by the desk machine (Pacific).
BOARD_TZ = ZoneInfo("America/Los_Angeles")
MAX_TOP = 15


@dataclass(frozen=True)
class Sources:
    """Where each section reads from; tests pass fakes, the app uses :func:`live_sources`."""

    snapshot: Callable[[], Mapping[str, Any] | None]
    industry_rows: Callable[[], list[Mapping[str, Any]]]
    sector_rows: Callable[[], list[Mapping[str, Any]]]
    #: ``{SYM: {"industry": .., "sector": ..}}``
    symbol_map: Callable[[], Mapping[str, Mapping[str, Any]]]
    #: ``{SYM: "LONG"|"SHORT"}`` open journal positions.
    book: Callable[[], Mapping[str, str]]
    #: Focus names (any category or side).
    focus: Callable[[], Iterable[str]]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _live_snapshot() -> Mapping[str, Any] | None:
    from project_paths import INDUSTRY_BOARD_STATE_FILE

    payload = _read_json(Path(INDUSTRY_BOARD_STATE_FILE))
    return payload if isinstance(payload, dict) else None


def _board_path(key: str, fallback_name: str) -> Path:
    snap = _live_snapshot() or {}
    if snap.get(key):
        return Path(str(snap[key]))
    import industry_scanner

    return Path(getattr(industry_scanner, fallback_name))


def _live_industry_rows() -> list[Mapping[str, Any]]:
    from industry_context import _read_csv_rows

    return _read_csv_rows(_board_path("industry_path", "INDUSTRY_BOARD_CSV_FILE"))


def _live_sector_rows() -> list[Mapping[str, Any]]:
    from industry_context import _read_csv_rows

    return _read_csv_rows(_board_path("sector_path", "SECTOR_BOARD_CSV_FILE"))


def _live_symbol_map() -> Mapping[str, Mapping[str, Any]]:
    from industry_context import load_industry_context_map

    return load_industry_context_map()


def _live_book() -> dict[str, str]:
    from mentor_packs import context_pack

    return {
        str(row.get("symbol") or "").strip().upper(): str(row.get("direction") or "").strip().upper()
        for row in context_pack._live_open_positions()
        if row.get("symbol")
    }


def _live_focus() -> list[str]:
    from mentor_packs import context_pack

    focus = context_pack._live_focus()
    return [sym for cat in focus.values() for names in cat.values() for sym in names]


def live_sources() -> Sources:
    return Sources(
        snapshot=_live_snapshot,
        industry_rows=_live_industry_rows,
        sector_rows=_live_sector_rows,
        symbol_map=_live_symbol_map,
        book=_live_book,
        focus=_live_focus,
    )


def _float(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _pct(value: Any) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:+.1f}%"


def _slug(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(label).strip().lower()).strip("_") or "unnamed"


def _board_time(text: str) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(text or "").strip())
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=BOARD_TZ)


def _previous_weekday(day: date) -> date:
    back = day - timedelta(days=1)
    while back.weekday() >= 5:
        back -= timedelta(days=1)
    return back


def _age_text(seconds: float) -> str:
    hours = seconds / 3600.0
    return f"{hours * 60:.0f} min" if hours < 1 else f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} days"


def _names_in(group: str, level: str, symbol_map: Mapping[str, Mapping[str, Any]], book: Mapping[str, str],
              liked: set[str], focus: set[str]) -> list[str]:
    """The trader's names in one group: book first (with side), then liked, then Focus."""
    key = "industry" if level == "industry" else "sector"
    wanted = str(group).strip().lower()
    aliases = {wanted}
    if level == "sector":
        from industry_context import _SECTOR_ALIASES

        aliases |= {raw for raw, board in _SECTOR_ALIASES.items() if board.lower() == wanted}
    tagged: list[tuple[int, str]] = []
    for sym in sorted(set(book) | liked | focus):
        if str((symbol_map.get(sym) or {}).get(key) or "").strip().lower() not in aliases:
            continue
        if sym in book:
            tagged.append((0, f"{sym} (book {book[sym] or '?'})"))
        elif sym in liked:
            tagged.append((1, f"{sym} (liked)"))
        else:
            tagged.append((2, f"{sym} (Focus)"))
    return [text for _, text in sorted(tagged)]


def day_label(stamp: datetime, session: date) -> str:
    """What the board's 1d column is: today so far when built in today's session, else the last session's change."""
    built = stamp.astimezone(ET)
    if built.date() == session and built.time() >= time(9, 30):
        return f"today so far (as of {built:%H:%M} ET)"
    return "1d (last session)"


def _group_row(row_id: str, side: str, row: Mapping[str, Any], level: str, names: list[str],
               one_day: str = "1d") -> dict[str, Any]:
    label = str(row.get(level) or "").strip()
    members = row.get("member_count")
    rank = _float(row.get("rs_rank"))
    rank_text = "?" if rank is None else f"{rank:g}"
    extra = f", {members} members" if members not in (None, "") else (f" ({row.get('etf')})" if row.get("etf") else "")
    text = (f"{side} {level} #{rank_text}: {label}{extra}, {one_day} {_pct(row.get('pct_change_1d'))}, "
            f"5d {_pct(row.get('return_5d_pct'))}; yours: {', '.join(names) if names else 'none'}")
    return {"id": row_id, "kind": f"rs_{side}", "level": level, "group": label, "rank": rank, "text": text}


def build(level: str = "industry", top: int = 5, liked: Any = (), *, now: datetime | None = None,
          sources: Sources | None = None) -> Pack:
    """Build the RS pack. File reads only: call it on a worker."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    level = "sector" if str(level or "").strip().lower().startswith("sector") else "industry"
    top = max(1, min(MAX_TOP, int(top or 5)))
    src = sources or live_sources()
    snapshot = src.snapshot() or {}
    stamp = _board_time(str(snapshot.get("last_success_at") or ""))
    if stamp is None:
        return make_pack(NAME, [{"id": "rs:none", "kind": "none",
                                 "text": "Industry board: no snapshot on disk; relative strength unknown"}])
    age = (moment - stamp).total_seconds()
    session = moment.astimezone(ET).date()
    board_day = stamp.astimezone(ET).date()
    if board_day < _previous_weekday(session):
        return make_pack(NAME, [{"id": "rs:none", "kind": "none",
                                 "text": (f"Industry board is stale: last built {stamp.astimezone(ET):%Y-%m-%d %H:%M} ET "
                                          f"({_age_text(age)} ago, more than one session); relative strength unknown")}])
    board = list(src.industry_rows() if level == "industry" else src.sector_rows())
    board = [row for row in board if _float(row.get("rs_rank")) is not None and str(row.get(level) or "").strip()]
    board.sort(key=lambda row: _float(row.get("rs_rank")) or 0.0)
    rows: list[dict[str, Any]] = [{
        "id": "rs:asof", "kind": "asof",
        "at_utc": stamp.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "text": (f"{level.title()} board built {stamp.astimezone(ET):%Y-%m-%d %H:%M} ET ({_age_text(age)} ago), "
                 f"{len(board)} {level} groups ranked by RS (1 = strongest)"),
    }]
    if not board:
        rows.append({"id": "rs:none", "kind": "none", "text": f"The {level} board file has no ranked rows; unknown"})
        return make_pack(NAME, rows)
    try:
        symbol_map = src.symbol_map() or {}
    except Exception:  # noqa: BLE001 - names unknown, the board still reads
        symbol_map = {}
    book = {str(k).upper(): str(v).upper() for k, v in (src.book() or {}).items()}
    likes = {str(item[0] if isinstance(item, (list, tuple)) else item).strip().upper() for item in liked or () if item}
    focus = {str(sym).strip().upper() for sym in src.focus() or () if sym}
    # Leaders and laggards never share a group: at most half the board on each side.
    top = max(1, min(top, len(board) // 2))
    one_day = day_label(stamp, session)
    leaders, laggards = board[:top],list(reversed(board[-top:])) if len(board) > top else []
    for n, row in enumerate(leaders, 1):
        rows.append(_group_row(f"rs:lead:{n}", "lead", row, level,
                               _names_in(row[level], level, symbol_map, book, likes, focus), one_day))
    for n, row in enumerate(laggards, 1):
        rows.append(_group_row(f"rs:lag:{n}", "lag", row, level,
                               _names_in(row[level], level, symbol_map, book, likes, focus), one_day))
    by_label = {str(row.get(level) or "").strip().lower(): row for row in board}
    key = "industry" if level == "industry" else "sector"
    touched: dict[str, list[str]] = {}
    for sym, side in book.items():
        group = str((symbol_map.get(sym) or {}).get(key) or "").strip()
        touched.setdefault(group, []).append(f"{sym} {side or '?'}")
    seen: set[str] = set()
    for group, syms in sorted(touched.items()):
        row = by_label.get(group.lower())
        if not group:
            row_id, text = "rs:you:unmapped", f"Book names with no {level} on the board: {', '.join(syms)}"
        elif row is None:
            row_id, text = f"rs:you:{_slug(group)}", f"Your {group} ({', '.join(syms)}): not ranked on the {level} board"
        else:
            row_id = f"rs:you:{_slug(group)}"
            text = (f"Your {group} ({', '.join(syms)}): rank {_float(row.get('rs_rank')):g} of {len(board)}, "
                    f"{one_day} {_pct(row.get('pct_change_1d'))}, 5d {_pct(row.get('return_5d_pct'))}")
        if row_id in seen:
            continue
        seen.add(row_id)
        rows.append({"id": row_id, "kind": "rs_you", "level": level, "group": group, "text": text})
    return make_pack(NAME, rows)


def fixture_sources(*, built: str = "2026-09-30T13:39:56") -> Sources:
    industries = [
        {"industry": "Cybersecurity", "member_count": "11", "pct_change_1d": "2.06", "return_5d_pct": "0.01", "rs_rank": "1"},
        {"industry": "Semiconductors", "member_count": "40", "pct_change_1d": "1.20", "return_5d_pct": "3.40", "rs_rank": "2"},
        {"industry": "Software", "member_count": "60", "pct_change_1d": "0.50", "return_5d_pct": "1.10", "rs_rank": "3"},
        {"industry": "Insurance", "member_count": "25", "pct_change_1d": "-0.80", "return_5d_pct": "-2.00", "rs_rank": "4"},
        {"industry": "Autos", "member_count": "9", "pct_change_1d": "-1.90", "return_5d_pct": "-4.20", "rs_rank": "5"},
    ]
    sectors = [
        {"etf": "XLK", "sector": "Technology", "pct_change_1d": "0.64", "return_5d_pct": "0.21", "rs_rank": "1"},
        {"etf": "XLE", "sector": "Energy", "pct_change_1d": "-0.06", "return_5d_pct": "-1.39", "rs_rank": "2"},
        {"etf": "XLF", "sector": "Financials", "pct_change_1d": "-0.50", "return_5d_pct": "-1.80", "rs_rank": "3"},
    ]
    symbol_map = {
        "NVDA": {"industry": "Semiconductors", "sector": "Technology"},
        "AMD": {"industry": "Semiconductors", "sector": "Technology"},
        "ALL": {"industry": "Insurance", "sector": "Financial Services"},
        "TSLA": {"industry": "Autos", "sector": "Consumer Cyclical"},
        "CRWD": {"industry": "Cybersecurity", "sector": "Technology"},
    }
    return Sources(
        snapshot=lambda: {"last_success_at": built, "status": "ok"},
        industry_rows=lambda: [dict(row) for row in industries],
        sector_rows=lambda: [dict(row) for row in sectors],
        symbol_map=lambda: symbol_map,
        book={"NVDA": "LONG", "ALL": "SHORT"}.copy,
        focus=lambda: ["AMD", "TSLA", "CRWD"],
    )


FIXTURE_NOW = datetime(2026, 9, 30, 21, 0, tzinfo=timezone.utc)


def fixture() -> Pack:
    return build(level="industry", top=2, liked=["CRWD"], now=FIXTURE_NOW, sources=fixture_sources())
