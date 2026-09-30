"""Per-symbol news headlines from two free RSS feeds (Yahoo Finance primary, Google News second).

Pure and Qt-free. A headline is title + URL + source + published time, never an article
body: nothing here fetches the linked page. An item without a URL or a title is dropped at
ingest; a published time in the future is clamped to ``now``; the same URL twice is one
headline. :class:`NewsFetcher` is the only caller that touches the network in the app: one
process-wide lock, at least 30 min per symbol between fetches, at most 40 symbols per
30-min cycle, a 10 s timeout per feed, and it never raises (an empty list and a reason).
"""

from __future__ import annotations

import calendar
import html
import logging
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from urllib.parse import parse_qsl, quote_plus, urlencode, urlsplit, urlunsplit

FEED_YAHOO = "yahoo"
FEED_GOOGLE = "google"
YAHOO_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
GOOGLE_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
USER_AGENT = "TradingBotV3 mentor news"
TIMEOUT_SECONDS = 10
MIN_INTERVAL = timedelta(minutes=30)
CYCLE_LENGTH = timedelta(minutes=30)
MAX_SYMBOLS_PER_CYCLE = 40
#: Newest items kept per feed per fetch (Google returns ~100).
MAX_ITEMS_PER_FEED = 25
MAX_BODY_BYTES = 2_000_000
MAX_TITLE_CHARS = 300
_TAG_RE = re.compile(r"<[^>]*>")
_SPACE_RE = re.compile(r"\s+")
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_TRACKING_PREFIXES = ("utm_",)
_TRACKING_KEYS = frozenset({"guccounter", "guce_referrer", "guce_referrer_sig", "ncid", ".tsrc"})
#: Process-wide: one feed request at a time, whoever asks.
_NET_LOCK = threading.Lock()


@dataclass(frozen=True)
class Headline:
    title: str
    url: str
    source: str
    #: UTC ISO time, or "" when the feed gave none (shown as "time unknown", never guessed).
    published_utc: str
    symbol: str
    feed: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def clean_symbol(value: Any) -> str:
    """An upper-case ticker, or "" when ``value`` is not one."""
    text = str(value or "").strip().upper()
    return text if _SYMBOL_RE.match(text) else ""


def feed_urls(symbol: str) -> list[tuple[str, str]]:
    """``[(feed, url)]`` for one ticker, primary first."""
    sym = clean_symbol(symbol)
    if not sym:
        return []
    return [
        (FEED_YAHOO, YAHOO_URL.format(symbol=quote_plus(sym))),
        (FEED_GOOGLE, GOOGLE_URL.format(query=quote_plus(f"{sym} stock"))),
    ]


def normalize_url(url: Any) -> str:
    """The URL used for dedupe and storage: http(s) only, lower-case host, no fragment or tracking keys."""
    text = str(url or "").strip()
    try:
        parts = urlsplit(text)
    except ValueError:
        return ""
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.netloc:
        return ""
    query = [
        (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith(_TRACKING_PREFIXES) and key.lower() not in _TRACKING_KEYS
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, parts.netloc.lower(), path, urlencode(query), ""))


def _plain(text: Any) -> str:
    """Tags stripped, entities decoded, whitespace collapsed; never parsed beyond that."""
    value = html.unescape(_TAG_RE.sub(" ", str(text or "")))
    return _SPACE_RE.sub(" ", value).strip()[:MAX_TITLE_CHARS]


def _domain(url: str) -> str:
    try:
        host = urlsplit(url).netloc.lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _published(entry: Any, now: datetime) -> str:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return ""
    try:
        moment = datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return ""
    return min(moment, now).isoformat(timespec="seconds")


def _source(entry: Any, url: str) -> str:
    """The publisher's domain: Google's ``<source url=...>`` when given, else the link's own host."""
    src = entry.get("source") or {}
    href = str(src.get("href") or "") if hasattr(src, "get") else ""
    return _domain(href) or _domain(url)


def _now(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.astimezone()


def parse_feed(body: bytes | str, *, symbol: str, feed: str, now: datetime | None = None) -> list[Headline]:
    """Headlines in one RSS/Atom body, newest first. Raises ValueError for a body with no feed in it."""
    import feedparser

    moment = _now(now).astimezone(timezone.utc)
    data = body.encode("utf-8") if isinstance(body, str) else bytes(body or b"")
    # Bytes, never a str: feedparser treats a str as a URL or a file name.
    parsed = feedparser.parse(data)
    entries = list(parsed.get("entries") or ())
    if not entries and parsed.get("bozo"):
        raise ValueError(f"malformed feed ({type(parsed.get('bozo_exception')).__name__})")
    out: list[Headline] = []
    seen: set[str] = set()
    for entry in entries:
        url = normalize_url(entry.get("link"))
        title = _plain(entry.get("title"))
        if not url or not title or url in seen:
            continue  # no URL or no title: dropped at ingest
        seen.add(url)
        out.append(Headline(title=title, url=url, source=_source(entry, url), published_utc=_published(entry, moment),
                            symbol=clean_symbol(symbol), feed=feed))
    out.sort(key=lambda item: item.published_utc, reverse=True)
    return out[:MAX_ITEMS_PER_FEED]


def _body(response: Any) -> bytes:
    status = int(getattr(response, "status_code", 200) or 200)
    if status != 200:
        raise ValueError(f"HTTP {status}")
    content = getattr(response, "content", None)
    if content is None:
        content = str(getattr(response, "text", "") or "").encode("utf-8")
    if len(content) > MAX_BODY_BYTES:
        raise ValueError(f"feed body over {MAX_BODY_BYTES} bytes")
    return bytes(content)


def fetch_symbol(
    symbol: str,
    *,
    now: datetime | None = None,
    get: Callable[..., Any] | None = None,
    errors: dict[str, str] | None = None,
    timeout: float = TIMEOUT_SECONDS,
) -> list[Headline]:
    """Both feeds for one ticker, merged, one headline per URL (the primary feed wins). Never raises.

    A failed feed leaves its reason in ``errors[feed]`` and the other feed still counts.
    """
    if get is None:
        import requests

        get = requests.get
    moment = _now(now)
    merged: list[Headline] = []
    seen: set[str] = set()
    for feed, url in feed_urls(symbol):
        try:
            response = get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            items = parse_feed(_body(response), symbol=symbol, feed=feed, now=moment)
        except Exception as exc:  # noqa: BLE001 - a dead feed is a reason, never a raise
            if errors is not None:
                errors[feed] = f"{type(exc).__name__}: {exc}"[:200]
            continue
        for item in items:
            if item.url not in seen:
                seen.add(item.url)
                merged.append(item)
    merged.sort(key=lambda item: item.published_utc, reverse=True)
    return merged


@dataclass
class FetchResult:
    symbol: str
    headlines: list[Headline] = field(default_factory=list)
    #: "" = fetched; else why nothing was fetched ("too soon", "cycle cap", "not a ticker").
    reason: str = ""
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def fetched(self) -> bool:
        return not self.reason


class NewsFetcher:
    """The one network caller: a lock, a per-symbol minimum interval and a per-cycle cap. Never raises."""

    def __init__(
        self,
        *,
        get: Callable[..., Any] | None = None,
        min_interval: timedelta = MIN_INTERVAL,
        max_per_cycle: int = MAX_SYMBOLS_PER_CYCLE,
        cycle_length: timedelta = CYCLE_LENGTH,
    ) -> None:
        self._get = get
        self.min_interval = min_interval
        self.max_per_cycle = int(max_per_cycle)
        self.cycle_length = cycle_length
        self._lock = threading.Lock()
        self._last: dict[str, datetime] = {}
        self._cycle_start: datetime | None = None
        self.cycle_count = 0
        #: Per feed: the last failure reason ("" after a success).
        self.last_error: dict[str, str] = {}

    def seed(self, symbol: str, when: datetime) -> None:
        """Remember a fetch made before a restart (the store's last-fetch stamp)."""
        sym = clean_symbol(symbol)
        if sym and when is not None:
            with self._lock:
                moment = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
                if moment > self._last.get(sym, moment - timedelta(seconds=1)):
                    self._last[sym] = moment

    def last_fetch(self, symbol: str) -> datetime | None:
        with self._lock:
            return self._last.get(clean_symbol(symbol))

    def due(self, symbol: str, now: datetime | None = None) -> bool:
        last = self.last_fetch(symbol)
        return last is None or _now(now) - last >= self.min_interval

    def fetch(self, symbol: str, *, now: datetime | None = None) -> FetchResult:
        sym = clean_symbol(symbol)
        moment = _now(now)
        if not sym:
            return FetchResult(str(symbol or ""), reason="not a ticker")
        # The checks and the stamp under the state lock; the request under the process-wide one.
        with self._lock:
            last = self._last.get(sym)
            if last is not None and moment - last < self.min_interval:
                return FetchResult(sym, reason="too soon (fetched under 30 min ago)")
            if self._cycle_start is None or moment - self._cycle_start >= self.cycle_length:
                self._cycle_start, self.cycle_count = moment, 0
            if self.cycle_count >= self.max_per_cycle:
                return FetchResult(sym, reason=f"cycle cap ({self.max_per_cycle} symbols per 30 min)")
            self.cycle_count += 1
            self._last[sym] = moment
            count = self.cycle_count
        errors: dict[str, str] = {}
        with _NET_LOCK:
            headlines = fetch_symbol(sym, now=moment, get=self._get, errors=errors)
        with self._lock:
            for feed, _ in feed_urls(sym):
                self.last_error[feed] = errors.get(feed, "")
        logging.info("Trade Mentor news: %s fetched (%d headlines; %d/%d symbols this cycle)%s",
                     sym, len(headlines), count, self.max_per_cycle, f" errors {errors}" if errors else "")
        return FetchResult(sym, headlines=headlines, errors=errors)
