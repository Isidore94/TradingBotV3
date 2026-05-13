from __future__ import annotations

import json
import logging
import os
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from market_prep.cache import get_default_cache_dir, read_json_cache, write_json_cache
from market_prep.config_loader import CONFIG_DIR
from market_prep.models import MarketPrepConfig

try:
    import feedparser
except ImportError:  # pragma: no cover - depends on optional local environment
    feedparser = None


NEWS_RSS_FEEDS_FILE = CONFIG_DIR / "news_rss_feeds.json"
RSS_NEWS_CACHE_FILE_NAME = "rss_news_cache.json"
FEEDPARSER_MISSING_MESSAGE = "feedparser not installed; using built-in RSS parser fallback."
GOOGLE_NEWS_BASE_URL = "https://news.google.com/rss/search"
DEFAULT_NEWS_RSS_CONFIG = {
    "feeds": [
        {
            "name": "Yahoo Finance",
            "url": "https://finance.yahoo.com/news/rssindex",
            "category": "market",
        }
    ]
}
DEFAULT_GOOGLE_NEWS_SETTINGS = {
    "enabled": False,
    "cache_ttl_hours": 3,
    "max_watchlist_tickers": 30,
    "max_feeds": 80,
    "request_timeout_seconds": 10,
    "queries": [
        "{ticker} earnings",
        "{ticker} guidance",
        "{ticker} downgrade",
        "{ticker} lawsuit",
        "{ticker} acquisition",
        "Federal Reserve inflation",
        "Treasury auction yields",
        "semiconductor export controls",
        "oil prices Middle East",
    ],
}

KEYWORD_BUCKETS = {
    "Fed/rates": ("fed", "fomc", "rate", "rates", "powell", "yield", "treasury"),
    "inflation": ("inflation", "cpi", "ppi", "pce", "prices"),
    "employment": ("jobs", "payroll", "payrolls", "unemployment", "employment", "jobless", "adp"),
    "geopolitics": ("geopolitical", "war", "conflict", "sanction", "ukraine", "russia", "israel", "iran"),
    "oil/energy": ("oil", "crude", "energy", "opec", "gasoline", "natural gas"),
    "China": ("china", "chinese", "beijing"),
    "AI/semis": ("ai", "artificial intelligence", "semiconductor", "semis", "chips", "nvidia", "nvda"),
    "banking/credit": ("bank", "banks", "credit", "lending", "default", "debt"),
    "tariffs/trade": ("tariff", "tariffs", "trade", "imports", "exports"),
    "earnings": ("earnings", "guidance", "revenue", "profit", "eps"),
    "regulation": ("regulation", "regulator", "sec", "doj", "ftc", "lawsuit"),
    "M&A": ("merger", "acquisition", "buyout", "deal"),
    "analyst": ("downgrade", "upgrade", "price target", "rating"),
}


def fetch_rss_headlines(
    *,
    limit: int = 25,
    config_path: Path = NEWS_RSS_FEEDS_FILE,
    config: MarketPrepConfig | None = None,
    tickers: list[str] | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    ensure_news_rss_config(config_path)
    generated_at = datetime.now().isoformat(timespec="seconds")
    cache_path = _cache_path(config)
    ttl = timedelta(hours=max(0.0, _safe_float(_google_news_settings(config).get("cache_ttl_hours"), 3.0)))
    cache_payload = read_json_cache(cache_path, default={})
    if not force_refresh and _is_cache_fresh(cache_payload, ttl):
        cached_headlines = cache_payload.get("headlines") if isinstance(cache_payload, dict) else []
        cached_headlines = cached_headlines if isinstance(cached_headlines, list) else []
        return _payload(
            generated_at,
            cached_headlines[: max(0, int(limit))],
            "" if cached_headlines else "No RSS headlines found.",
            status="cache",
            status_label="Loaded from cache",
            warnings=list(cache_payload.get("warnings") or []) if isinstance(cache_payload, dict) else [],
            cache_path=cache_path,
        )

    feeds = _configured_feeds(config_path, config=config, tickers=tickers)
    headlines: list[dict[str, Any]] = []
    warnings: list[str] = []
    if feedparser is None:
        warnings.append(FEEDPARSER_MISSING_MESSAGE)

    for feed in feeds:
        name = str(feed.get("name") or "RSS Feed").strip()
        url = str(feed.get("url") or "").strip()
        category = str(feed.get("category") or "").strip()
        query = str(feed.get("query") or "").strip()
        if not url:
            warnings.append(f"Skipped RSS feed with blank URL: {name}")
            continue
        try:
            entries = _parse_feed_entries(url, timeout=_feed_timeout(config))
        except Exception as exc:
            logging.getLogger("market_prep").warning("RSS feed failed for %s: %s", url, exc)
            warnings.append(f"{name}: {exc}")
            continue
        for entry in entries:
            title = str(entry.get("title") or "").strip()
            if not title:
                continue
            summary = str(entry.get("summary") or "").strip()
            tags = tag_headline(title + " " + summary + " " + query)
            headlines.append(
                {
                    "title": title,
                    "source": name,
                    "published": str(entry.get("published") or "").strip(),
                    "url": str(entry.get("url") or "").strip(),
                    "category": category,
                    "query": query,
                    "summary": summary,
                    "tags": tags,
                }
            )

    deduped = _dedupe_headlines(headlines)
    limited = deduped[: max(0, int(limit))]
    message = "" if limited else "No RSS headlines found."
    payload = _payload(
        generated_at,
        limited,
        message,
        status="refreshed",
        status_label="Refreshed",
        warnings=warnings,
        cache_path=cache_path,
    )
    _write_cache(payload, cache_path=cache_path)
    return payload


def load_news_rss_feeds(path: Path = NEWS_RSS_FEEDS_FILE) -> list[dict[str, Any]]:
    ensure_news_rss_config(path)
    payload = _read_json(path, DEFAULT_NEWS_RSS_CONFIG)
    feeds = payload.get("feeds") if isinstance(payload, dict) else []
    return [feed for feed in feeds if isinstance(feed, dict)] if isinstance(feeds, list) else []


def ensure_news_rss_config(path: Path = NEWS_RSS_FEEDS_FILE) -> bool:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return False
    _write_json_atomic(target, DEFAULT_NEWS_RSS_CONFIG)
    return True


def tag_headline(text: str) -> list[str]:
    lowered = str(text or "").lower()
    tags = []
    for bucket, keywords in KEYWORD_BUCKETS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            tags.append(bucket)
    return tags


def _configured_feeds(
    config_path: Path,
    *,
    config: MarketPrepConfig | None,
    tickers: list[str] | None,
) -> list[dict[str, Any]]:
    feeds: list[dict[str, Any]] = []
    if config is None or bool(config.features.get("news", False)):
        feeds.extend(load_news_rss_feeds(config_path))
    feeds.extend(_google_news_feeds(config, tickers=tickers))
    max_feeds = max(1, _safe_int(_google_news_settings(config).get("max_feeds"), 80))
    return feeds[:max_feeds]


def _google_news_feeds(config: MarketPrepConfig | None, *, tickers: list[str] | None) -> list[dict[str, Any]]:
    settings = _google_news_settings(config)
    if not bool(settings.get("enabled", False)):
        return []
    queries = settings.get("queries") if isinstance(settings.get("queries"), list) else []
    max_tickers = max(0, _safe_int(settings.get("max_watchlist_tickers"), 30))
    watchlist_tickers = _normalize_tickers(tickers or [])[:max_tickers]
    feeds: list[dict[str, Any]] = []
    seen_queries = set()
    for query_template in queries:
        template = str(query_template or "").strip()
        if not template:
            continue
        if "{ticker}" in template:
            for ticker in watchlist_tickers:
                query = template.replace("{ticker}", ticker)
                _append_google_feed(feeds, query, seen_queries)
        else:
            _append_google_feed(feeds, template, seen_queries)
    return feeds


def _append_google_feed(feeds: list[dict[str, Any]], query: str, seen_queries: set[str]) -> None:
    cleaned = " ".join(str(query or "").split())
    if not cleaned:
        return
    key = cleaned.lower()
    if key in seen_queries:
        return
    seen_queries.add(key)
    params = {
        "q": cleaned,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    feeds.append(
        {
            "name": f"Google News: {cleaned}",
            "url": GOOGLE_NEWS_BASE_URL + "?" + urllib.parse.urlencode(params),
            "category": "google_news",
            "query": cleaned,
        }
    )


def _parse_feed_entries(url: str, *, timeout: int) -> list[dict[str, Any]]:
    if feedparser is not None:
        parsed = feedparser.parse(url)
        entries = getattr(parsed, "entries", []) or []
        return [
            {
                "title": str(entry.get("title") or "").strip(),
                "url": str(entry.get("link") or "").strip(),
                "published": str(entry.get("published") or entry.get("updated") or "").strip(),
                "summary": str(entry.get("summary") or "").strip(),
            }
            for entry in entries
        ]
    return _parse_feed_entries_xml(url, timeout=timeout)


def _parse_feed_entries_xml(url: str, *, timeout: int) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TradingBotV3 market prep",
            "Accept": "application/rss+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace").lstrip("\ufeff")
    root = ET.fromstring(text)
    entries = []
    for item in root.findall(".//item"):
        entries.append(
            {
                "title": _xml_text(item, "title"),
                "url": _xml_text(item, "link"),
                "published": _xml_text(item, "pubDate") or _xml_text(item, "updated"),
                "summary": _xml_text(item, "description"),
            }
        )
    return entries


def _xml_text(item: ET.Element, tag: str) -> str:
    node = item.find(tag)
    return (node.text or "").strip() if node is not None else ""


def _dedupe_headlines(headlines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen = set()
    for headline in headlines:
        title = " ".join(str(headline.get("title") or "").lower().split())
        url = str(headline.get("url") or "").strip().lower()
        key = url or title
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(headline)
    deduped.sort(key=_headline_sort_key)
    return deduped


def _headline_sort_key(headline: dict[str, Any]) -> tuple[int, str, str]:
    tags = headline.get("tags") if isinstance(headline.get("tags"), list) else []
    priority = 0 if tags else 1
    return (priority, str(headline.get("published") or ""), str(headline.get("title") or ""))


def _payload(
    generated_at: str,
    headlines: list[dict[str, Any]],
    message: str,
    *,
    status: str = "",
    status_label: str = "",
    warnings: list[str] | None = None,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "source": "rss+google_news",
        "headlines": headlines,
        "message": message,
        "warnings": warnings or [],
        "status": status,
        "status_label": status_label,
        "cache_path": str(cache_path or ""),
    }


def _cache_path(config: MarketPrepConfig | None) -> Path:
    if config is not None:
        cache_dir = config.resolved_paths().get("cache_dir")
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir / RSS_NEWS_CACHE_FILE_NAME
    return get_default_cache_dir() / RSS_NEWS_CACHE_FILE_NAME


def _write_cache(payload: dict[str, Any], *, cache_path: Path | None = None) -> None:
    target = cache_path or get_default_cache_dir() / RSS_NEWS_CACHE_FILE_NAME
    write_json_cache(target, {**payload, "fetched_at": payload.get("generated_at")})


def _is_cache_fresh(payload: Any, ttl: timedelta) -> bool:
    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("headlines"), list):
        return False
    fetched = _parse_datetime(payload.get("fetched_at") or payload.get("generated_at"))
    if fetched is None:
        return False
    return datetime.now() - fetched <= ttl


def _google_news_settings(config: MarketPrepConfig | None) -> dict[str, Any]:
    settings = dict(DEFAULT_GOOGLE_NEWS_SETTINGS)
    if config is not None and isinstance(config.google_news_rss, dict):
        settings.update(config.google_news_rss)
    return settings


def _feed_timeout(config: MarketPrepConfig | None) -> int:
    settings = _google_news_settings(config)
    return max(1, _safe_int(settings.get("request_timeout_seconds"), 10))


def _normalize_tickers(values: list[str]) -> list[str]:
    tickers: list[str] = []
    seen = set()
    for value in values:
        ticker = str(value or "").strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def _parse_datetime(value: Any) -> datetime | None:
    try:
        text = str(value or "").strip()
        return datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
