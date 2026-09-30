"""News pack: one symbol's stored headlines, each with its URL, source and published time. Read-only.

Rows ``news:<SYM>:<n>`` newest first (``n`` is the headline's row id in the app's own
chat store, so an id always means the same headline), at most 8; ``news:<SYM>:none``
when there are none; ``news:<SYM>:asof`` with the window and the last fetch. A headline
row always carries its URL in its text: a headline without a URL is never a row. The
pack never fetches (the app's news job does) and never reads an article body.

The model may cite a headline id; the citation check drops any reply citing an id the
pack did not carry. A quoted headline string is not checked separately: the citation
rule is enough, because every cited id renders with its own title and URL.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.registry import Pack, make_pack

NAME = "news_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Recent news headlines for one stock (Yahoo Finance and Google News RSS): title, source, "
            "published time and URL. Headlines only, never the article text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker, e.g. NVDA."},
                "days": {"type": "integer", "description": "How many days back (1-14, default 3)."},
            },
            "required": ["symbol"],
        },
    },
}

PT = ZoneInfo("America/Los_Angeles")
DEFAULT_DAYS = 3
MAX_DAYS = 14
MAX_ROWS = 8
#: Newest first by published time (the fetch time when the feed gave none).
NEWS_SELECT = (
    "SELECT id, symbol, title, url, source, published_utc, fetched_utc, feed FROM news "
    "WHERE symbol = ? AND url <> '' AND COALESCE(NULLIF(published_utc, ''), fetched_utc) >= ? "
    "ORDER BY COALESCE(NULLIF(published_utc, ''), fetched_utc) DESC, id DESC LIMIT ?"
)
FETCH_KEY = "news:last_fetch:{symbol}"

#: (symbol, since UTC ISO, limit) -> stored rows, newest first.
Reader = Callable[[str, str, int], Iterable[Mapping[str, Any]]]
#: symbol -> the last fetch UTC ISO, or None when never fetched.
StampReader = Callable[[str], "str | None"]


def _sym(value: Any) -> str:
    return str(value or "").strip().upper()


def _now(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.astimezone()


def _connect_ro(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def db_reader(path: Path | str) -> Reader:
    """Rows from the chat store opened ``mode=ro``; a missing file or table is no rows."""
    target = Path(path)

    def read(symbol: str, since: str, limit: int) -> list[dict[str, Any]]:
        conn = _connect_ro(target)
        if conn is None:
            return []
        try:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='news'").fetchone() is None:
                return []
            return [dict(row) for row in conn.execute(NEWS_SELECT, (_sym(symbol), since, int(limit)))]
        finally:
            conn.close()

    return read


def db_stamp_reader(path: Path | str) -> StampReader:
    target = Path(path)

    def read(symbol: str) -> str | None:
        conn = _connect_ro(target)
        if conn is None:
            return None
        try:
            row = conn.execute("SELECT value FROM app_state WHERE key = ?", (FETCH_KEY.format(symbol=_sym(symbol)),))
            found = row.fetchone()
            return str(found["value"]) if found else None
        except sqlite3.Error:
            return None
        finally:
            conn.close()

    return read


def live_reader() -> Reader:
    from project_paths import MENTOR_CHAT_DB_FILE

    return db_reader(MENTOR_CHAT_DB_FILE)


def live_stamp_reader() -> StampReader:
    from project_paths import MENTOR_CHAT_DB_FILE

    return db_stamp_reader(MENTOR_CHAT_DB_FILE)


def when_text(value: Any) -> str:
    """A UTC ISO time as ``Wed 09-30 07:43 PT``; "time unknown" when there is none."""
    try:
        moment = datetime.fromisoformat(str(value or ""))
    except ValueError:
        return "time unknown"
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return f"{moment.astimezone(PT):%a %m-%d %H:%M} PT"


def headline_rows(symbol: str, *, now: datetime, days: int, limit: int, reader: Reader) -> list[dict[str, Any]]:
    """The citable headline rows (``news:<SYM>:<n>``); a stored row without an http(s) URL is skipped."""
    sym = _sym(symbol)
    since = (now - timedelta(days=days)).astimezone(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    for item in reader(sym, since, int(limit)) or ():
        url, title = str(item.get("url") or "").strip(), str(item.get("title") or "").strip()
        if not title or not url.startswith(("http://", "https://")):
            continue
        source = str(item.get("source") or "").strip() or "unknown source"
        published = str(item.get("published_utc") or "")
        rows.append({
            "id": f"news:{sym}:{item.get('id')}",
            "kind": "news",
            "title": title,
            "url": url,
            "source": source,
            "published_utc": published,
            "text": f'Headline "{title}" ({source}, {when_text(published)}) {url}',
        })
    return rows[:limit]


def clamp_days(days: Any) -> int:
    try:
        value = int(days)
    except (TypeError, ValueError):
        value = DEFAULT_DAYS
    return max(1, min(MAX_DAYS, value))


def build(
    symbol: str = "",
    days: Any = DEFAULT_DAYS,
    *,
    now: datetime | None = None,
    reader: Reader | None = None,
    stamps: StampReader | None = None,
) -> Pack:
    """Build the news pack for ``symbol`` from the stored headlines. A DB read: call it on a worker."""
    sym = _sym(symbol)
    if not sym or not sym.replace(".", "").replace("-", "").isalnum():
        return make_pack(NAME, (), empty_text="news_pack needs a ticker, e.g. NVDA")
    window = clamp_days(days)
    moment = _now(now)
    reader = reader or live_reader()
    stamps = stamps or live_stamp_reader()
    try:
        fetched = stamps(sym)
    except Exception:  # noqa: BLE001 - the stamp is a label; unreadable = unknown
        fetched = None
    last = f"last fetched {when_text(fetched)}" if fetched else "never fetched"
    rows: list[dict[str, Any]] = [{
        "id": f"news:{sym}:asof",
        "kind": "asof",
        "fetched_utc": fetched or "",
        "text": f"News for {sym}, last {window} day{'s' if window != 1 else ''} (Yahoo Finance / Google News RSS; {last})",
    }]
    try:
        found = headline_rows(sym, now=moment, days=window, limit=MAX_ROWS, reader=reader)
    except Exception as exc:  # noqa: BLE001 - an unreadable store is unknown, never "no news"
        return make_pack(NAME, rows + [{"id": f"news:{sym}:none", "kind": "unknown",
                                        "text": f"Headlines: unknown ({type(exc).__name__})"}])
    if not found:
        why = "none stored" if fetched else "not fetched yet"
        found = [{"id": f"news:{sym}:none", "kind": "news_empty", "text": f"Headlines: {why} in the last {window} days"}]
    return make_pack(NAME, rows + found)


# ---------------------------------------------------------------- fixture
FIXTURE_NOW = datetime(2026, 9, 29, 14, 0, tzinfo=timezone.utc)  # Tue 2026-09-29, 07:00 PT
FIXTURE_HEADLINES: tuple[dict[str, Any], ...] = (
    {"id": 11, "symbol": "NVDA", "title": "Nvidia adds $150 billion buyback", "url": "https://www.nytimes.com/nvda-buyback",
     "source": "nytimes.com", "published_utc": "2026-09-28T13:49:59+00:00", "fetched_utc": "2026-09-29T13:30:00+00:00",
     "feed": "google"},
    {"id": 12, "symbol": "NVDA", "title": "Broadcom reports tomorrow; chips mixed", "url": "https://finance.yahoo.com/news/avgo",
     "source": "finance.yahoo.com", "published_utc": "2026-09-29T12:10:00+00:00",
     "fetched_utc": "2026-09-29T13:30:00+00:00", "feed": "yahoo"},
    {"id": 5, "symbol": "NVDA", "title": "Old story outside the window", "url": "https://example.com/old",
     "source": "example.com", "published_utc": "2026-09-20T12:00:00+00:00", "fetched_utc": "2026-09-20T13:00:00+00:00",
     "feed": "yahoo"},
    {"id": 13, "symbol": "TSLA", "title": "Tesla deliveries preview", "url": "https://www.reuters.com/tsla",
     "source": "reuters.com", "published_utc": "2026-09-29T11:00:00+00:00", "fetched_utc": "2026-09-29T13:30:00+00:00",
     "feed": "google"},
)
FIXTURE_FETCHED = "2026-09-29T13:30:00+00:00"


def list_reader(items: Iterable[Mapping[str, Any]]) -> Reader:
    """A reader over in-memory rows with the store's order and window (tests and fixtures)."""
    stored = [dict(item) for item in items]

    def read(symbol: str, since: str, limit: int) -> list[dict[str, Any]]:
        def when(item: Mapping[str, Any]) -> str:
            return str(item.get("published_utc") or item.get("fetched_utc") or "")

        mine = [item for item in stored if _sym(item.get("symbol")) == _sym(symbol) and item.get("url") and when(item) >= since]
        mine.sort(key=lambda item: (when(item), int(item.get("id") or 0)), reverse=True)
        return mine[: int(limit)]

    return read


def fixture_reader() -> Reader:
    return list_reader(FIXTURE_HEADLINES)


def fixture_stamps(symbol: str) -> str | None:
    return FIXTURE_FETCHED if _sym(symbol) in ("NVDA", "TSLA") else None


def fixture() -> Pack:
    return build("NVDA", now=FIXTURE_NOW, reader=fixture_reader(), stamps=fixture_stamps)
