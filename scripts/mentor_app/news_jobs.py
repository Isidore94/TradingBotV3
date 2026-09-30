"""News refresh jobs (P7): who gets fetched, when, and the /news card. Qt-free; no model.

Every 30 min, 06:00-13:30 PT on weekdays, the app fetches headlines for its scope: the
symbols the trader named (``/news``, ``/pick``, ``/check``), then the open journal
book, then the liked chips (at most 24), never all of Focus. ``news_feed.NewsFetcher``
caps a cycle at 40 symbols and a symbol at one fetch per 30 min. News is not the GPU,
so it runs while AI is paused; the window never runs it on the Qt thread, never while
the desk is closed, and it never posts to the Inbox or moves the transcript: headlines
only show inside pick/check cards and ``/news``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
SESSION_START = time(6, 0)
SESSION_END = time(13, 30)
REFRESH_EVERY = timedelta(minutes=30)
MAX_LIKED = 24
DEFAULT_DAYS = 3


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
) -> list[str]:
    """Named first, then the open book, then the first ``max_liked`` liked chips; each symbol once."""
    from news_feed import clean_symbol

    out: list[str] = []
    liked_list = [clean_symbol(sym) for sym in liked][: int(max_liked)]
    for sym in [*(clean_symbol(s) for s in named), *(clean_symbol(s) for s in open_book), *liked_list]:
        if sym and sym not in out:
            out.append(sym)
    return out


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
    """Fetch one symbol (the fetcher's gates apply) and keep new headlines. Never raises into the queue."""
    result = fetcher.fetch(symbol, now=now)
    added = 0
    if result.fetched:
        stamp = now.astimezone(timezone.utc).isoformat(timespec="seconds")
        added = store.put_headlines(result.headlines, stamp) or 0
        store.set_news_fetched(result.symbol, stamp)
    return {"symbol": result.symbol, "fetched": result.fetched, "reason": result.reason,
            "errors": dict(result.errors), "added": added}


def run_cycle(symbols: Iterable[str], *, store: Any, fetcher: Any, now: Callable[[], datetime],
              should_stop: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    """One cycle over ``symbols`` in order; logs how many were fetched against the cap."""
    fetched, skipped, added = [], [], 0
    for symbol in symbols:
        if should_stop():
            break
        out = refresh_symbol(symbol, store=store, fetcher=fetcher, now=now())
        if out["fetched"]:
            fetched.append(out["symbol"])
            added += int(out["added"])
        else:
            skipped.append((out["symbol"], out["reason"]))
    logging.info("Trade Mentor news cycle: %d symbols fetched (cap %d per 30 min), %d new headlines, %d skipped",
                 len(fetched), getattr(fetcher, "max_per_cycle", 0), added, len(skipped))
    return {"fetched": fetched, "skipped": skipped, "added": added}


def news_card(symbol: str, days: int = DEFAULT_DAYS, *, store: Any, fetcher: Any, now: datetime,
              build: Callable[..., Any] | None = None) -> dict[str, Any]:
    """``/news SYM``: fetch once when never fetched, then the card from the store. No model."""
    from mentor_packs import news_pack

    sym = str(symbol or "").strip().upper()
    note = ""
    if store.news_fetched(sym) is None:
        out = refresh_symbol(sym, store=store, fetcher=fetcher, now=now)
        if not out["fetched"]:
            note = f"not fetched: {out['reason']}"
        elif out["errors"] and not out["added"]:
            note = "fetch errors: " + "; ".join(f"{k}: {v}" for k, v in sorted(out["errors"].items()))
    build = build or news_pack.build
    pack = build(sym, days, now=now, reader=store.headlines, stamps=store.news_fetched)
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
