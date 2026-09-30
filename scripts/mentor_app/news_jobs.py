"""News refresh jobs (P7): who gets fetched, when, and the /news card. Qt-free; no model.

Every 30 min, 06:00-13:30 PT on weekdays, the app fetches headlines for its scope: the
symbols the trader named today (``/news``, ``/pick``, ``/check``; the 10 newest), then
the open journal book, then the liked chips (at most 24), never all of Focus; at most 40.
``news_feed.NewsFetcher`` caps a cycle at 40 symbols and a symbol at one request per 30
min. News is not the GPU, so it runs while AI is paused; the window runs it on its own
news thread (never the Qt thread, never the model queue), never while the desk is
closed, and it never posts to the Inbox or moves the transcript: headlines only show
inside pick/check cards and ``/news``. A request where every feed failed is "unknown".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
SESSION_START = time(6, 0)
SESSION_END = time(13, 30)
REFRESH_EVERY = timedelta(minutes=30)
MAX_LIKED = 24
#: Symbols the trader typed today, newest first; dropped when the PT day ends.
MAX_NAMED = 10
MAX_SCOPE = 40
DEFAULT_DAYS = 3


class NamedSymbols:
    """The symbols the trader typed (/news, /pick, /check) today: newest first, at most 10."""

    def __init__(self, cap: int = MAX_NAMED) -> None:
        self.cap = int(cap)
        self._items: list[tuple[str, Any]] = []  # (SYMBOL, PT day), newest first

    def add(self, symbol: str, now: datetime) -> None:
        from news_feed import clean_symbol

        sym = clean_symbol(symbol)
        if not sym:
            return
        day = now.astimezone(PT).date()
        self._items = [(sym, day)] + [item for item in self._items if item[0] != sym]
        del self._items[self.cap:]

    def today(self, now: datetime) -> list[str]:
        day = now.astimezone(PT).date()
        self._items = [item for item in self._items if item[1] == day]
        return [sym for sym, _ in self._items]

    def __contains__(self, symbol: object) -> bool:
        return any(sym == symbol for sym, _ in self._items)


@dataclass
class NewsSchedule:
    """Every 30 min inside 06:00-13:30 PT, weekdays. Marked only when a cycle is queued."""

    last_utc: datetime | None = None

    def due(self, now: datetime) -> bool:
        moment = now if now.tzinfo else now.astimezone()
        local = moment.astimezone(PT)
        if local.weekday() >= 5 or not (SESSION_START <= local.time() < SESSION_END):
            return False
        return self.last_utc is None or moment - self.last_utc >= REFRESH_EVERY

    def mark(self, now: datetime) -> None:
        moment = now if now.tzinfo else now.astimezone()
        self.last_utc = moment.astimezone(timezone.utc)


def news_scope(
    *,
    named: Iterable[str] = (),
    open_book: Iterable[str] = (),
    liked: Iterable[str] = (),
    max_liked: int = MAX_LIKED,
    max_named: int = MAX_NAMED,
    limit: int = MAX_SCOPE,
) -> list[str]:
    """The first ``max_named`` named (newest first), the open book, the first ``max_liked`` chips; at most
    ``limit`` symbols, each once. With 10 named and 24 chips, the open book always keeps 6 of the 40."""
    from news_feed import clean_symbol

    def unique(values: Iterable[str]) -> list[str]:
        seen: list[str] = []
        for value in values:
            sym = clean_symbol(value)
            if sym and sym not in seen:
                seen.append(sym)
        return seen

    named_list = unique(named)[: int(max_named)]
    liked_list = unique(liked)[: int(max_liked)]
    fixed = set(named_list) | set(liked_list)
    # A big open book gives up its tail; the named names and the chips keep their share.
    book = [sym for sym in unique(open_book) if sym not in fixed][: max(0, int(limit) - len(fixed))]
    return unique([*named_list, *book, *liked_list])[: int(limit)]


def seed_fetcher(fetcher: Any, store: Any) -> int:
    """Give the fetcher the store's last-fetch stamps (the 30-min spacing survives a restart)."""
    count = 0
    for symbol, stamp in (store.news_fetch_stamps() or {}).items():
        try:
            when = datetime.fromisoformat(stamp)
        except (TypeError, ValueError):
            continue
        fetcher.seed(symbol, when if when.tzinfo else when.replace(tzinfo=timezone.utc))
        count += 1
    return count


def refresh_symbol(symbol: str, *, store: Any, fetcher: Any, now: datetime) -> dict[str, Any]:
    """Fetch one symbol (the fetcher's gates apply) and keep new headlines. Never raises into the queue.

    Every request stamps ``last_attempt`` (the spacing). Only a fetch where a feed answered stamps
    ``last_fetch``; a fetch where every feed failed stamps nothing else and keeps ``last_error``, so
    the news reads "unknown" (never "none") and the next cycle tries again.
    """
    result = fetcher.fetch(symbol, now=now)
    added = 0
    stamp = now.astimezone(timezone.utc).isoformat(timespec="seconds")
    if result.attempted:
        store.set_news_attempt(result.symbol, stamp)
    if result.fetched:
        added = store.put_headlines(result.headlines, stamp) or 0
        store.set_news_fetched(result.symbol, stamp)
        partial = "; ".join(f"{feed} failed: {why}" for feed, why in sorted(result.errors.items()))
        store.set_news_error(result.symbol, partial, stamp, partial=True)  # "" clears an old failure
    elif result.failed:
        store.set_news_error(result.symbol, result.reason, stamp)
    return {"symbol": result.symbol, "fetched": result.fetched, "failed": result.failed, "reason": result.reason,
            "errors": dict(result.errors), "added": added}


def needs_first_fetch(store: Any, symbol: str) -> bool:
    """/news on a symbol with no good fetch yet fetches once (on the news thread) before its card."""
    return store.news_fetched(str(symbol or "").strip().upper()) is None


def fetch_note(out: Mapping[str, Any]) -> str:
    if out.get("fetched"):
        return ""
    if out.get("failed"):
        return ""  # the card's own "news unknown: last fetch failed" row says it
    return f"not fetched now: {out.get('reason')}"


def news_card(symbol: str, days: int = DEFAULT_DAYS, *, store: Any, now: datetime, note: str = "",
              build: Callable[..., Any] | None = None) -> dict[str, Any]:
    """``/news SYM``: the card from the store (no network, no model)."""
    from mentor_packs import news_pack

    sym = str(symbol or "").strip().upper()
    build = build or news_pack.build
    pack = build(sym, days, now=now, reader=store.headlines, stamps=store.news_fetched, errors=store.news_error)
    return {"symbol": sym, "pack": pack, "markdown": card_markdown(pack, note=note)}


def card_markdown(pack: Any, *, note: str = "") -> str:
    """The /news card: each headline as a link with source and time; a headline without a URL never shows."""
    rows = list(getattr(pack, "rows", ()) or ())
    if not rows:
        return f"**News**: {getattr(pack, 'empty_text', '') or 'nothing'}"
    head = next((row for row in rows if row.get("kind") == "asof"), None)
    lines = [f"**{head['text'] if head else 'News'}**", ""]
    shown = 0
    for row in rows:
        if row.get("kind") == "news":
            url = str(row.get("url") or "")
            if not url.startswith(("http://", "https://")):
                continue
            from mentor_packs.news_pack import when_text

            title = str(row.get("title") or "").replace("[", "(").replace("]", ")")
            lines.append(f"- [{title}]({url}) — {row.get('source')}, {when_text(row.get('published_utc'))} "
                         f"`[{row['id']}]`")
            shown += 1
        elif row.get("kind") != "asof":
            lines.append(f"- {row.get('text', '')}")
    if note:
        lines += ["", f"*({note})*"]
    lines += ["", "*Headlines only; open a link to read the story.*"]
    return "\n".join(lines)


def news_links(refs: Iterable[str], pack: Any) -> list[tuple[str, str]]:
    """(title, url) for every cited headline row in ``pack``: a cited headline always shows its URL."""
    by_id = {str(row.get("id")): row for row in getattr(pack, "rows", ()) or ()}
    out: list[tuple[str, str]] = []
    for ref in refs:
        row = by_id.get(str(ref))
        if row is not None and row.get("kind") == "news" and str(row.get("url") or "").startswith(("http://", "https://")):
            out.append((str(row.get("title") or row.get("source") or "link"), str(row["url"])))
    return out
