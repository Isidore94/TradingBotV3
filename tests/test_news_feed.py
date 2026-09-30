"""news_feed: per-symbol RSS headlines (Yahoo primary, Google second), URL-only, never raising."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import news_feed  # noqa: E402

NOW = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)

YAHOO_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Yahoo! Finance: NVDA News</title>
<item><title>Nvidia &amp; AMD rally on &lt;b&gt;chip&lt;/b&gt; deal</title>
<link>https://finance.yahoo.com/news/nvidia-amd-rally.html?utm_source=rss&amp;.tsrc=rss</link>
<pubDate>Wed, 30 Sep 2026 14:30:00 +0000</pubDate></item>
<item><title>No link here</title><pubDate>Wed, 30 Sep 2026 14:00:00 +0000</pubDate></item>
<item><title></title><link>https://example.com/untitled</link></item>
<item><title>From the future</title><link>https://www.fool.com/future-story/</link>
<pubDate>Thu, 01 Oct 2026 09:00:00 +0000</pubDate></item>
<item><title>Same story again</title><link>https://finance.yahoo.com/news/nvidia-amd-rally.html</link>
<pubDate>Wed, 30 Sep 2026 14:31:00 +0000</pubDate></item>
<item><title>Bad scheme</title><link>javascript:alert(1)</link></item>
<item><title>Undated story</title><link>https://example.com/undated</link></item>
</channel></rss>"""

GOOGLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>"NVDA stock" - Google News</title>
<item><title>Nvidia adds $150 billion buyback - The New York Times</title>
<link>https://news.google.com/rss/articles/CBMiABC?oc=5</link>
<pubDate>Mon, 28 Sep 2026 13:49:59 GMT</pubDate>
<source url="https://www.nytimes.com">The New York Times</source></item>
<item><title>Duplicate of the Yahoo one</title>
<link>https://finance.yahoo.com/news/nvidia-amd-rally.html</link>
<pubDate>Wed, 30 Sep 2026 14:40:00 GMT</pubDate></item>
</channel></rss>"""

ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>t</title>
<entry><title>Atom headline</title><link href="https://example.org/a"/>
<updated>2026-09-30T12:00:00Z</updated></entry></feed>"""

MALFORMED = "<html><body>not a feed <<<"


class _Resp:
    def __init__(self, body: str, status: int = 200) -> None:
        self.content = body.encode("utf-8")
        self.status_code = status


def _get(bodies: dict[str, object], calls: list | None = None):
    def get(url, **kwargs):
        if calls is not None:
            calls.append((url, kwargs))
        for key, body in bodies.items():
            if key in url:
                if isinstance(body, BaseException):
                    raise body
                return body if isinstance(body, _Resp) else _Resp(str(body))
        raise AssertionError(url)

    return get


def test_yahoo_parse_drops_items_without_url_or_title_and_clamps_the_future():
    items = news_feed.parse_feed(YAHOO_RSS, symbol="nvda", feed="yahoo", now=NOW)
    titles = [item.title for item in items]
    assert "No link here" not in titles and "Bad scheme" not in titles
    assert all(item.url.startswith("https://") and item.title for item in items)
    future = next(item for item in items if item.title == "From the future")
    assert future.published_utc == NOW.isoformat(timespec="seconds")
    assert future.source == "fool.com" and future.url == "https://www.fool.com/future-story"
    first = next(item for item in items if item.title.startswith("Nvidia"))
    assert first.title == "Nvidia & AMD rally on chip deal"
    assert first.url == "https://finance.yahoo.com/news/nvidia-amd-rally.html", "tracking keys stripped"
    assert "Same story again" not in titles, "one headline per normalized URL"
    undated = next(item for item in items if item.title == "Undated story")
    assert undated.published_utc == "" and items[-1] is undated
    assert {item.symbol for item in items} == {"NVDA"}


def test_google_parse_takes_the_publisher_domain_and_tz_aware_times():
    items = news_feed.parse_feed(GOOGLE_RSS, symbol="NVDA", feed="google", now=NOW)
    nyt = next(item for item in items if "buyback" in item.title)
    assert nyt.source == "nytimes.com" and nyt.feed == "google"
    assert datetime.fromisoformat(nyt.published_utc).tzinfo is not None


def test_atom_parses_too():
    items = news_feed.parse_feed(ATOM, symbol="X", feed="yahoo", now=NOW)
    assert [(i.title, i.url) for i in items] == [("Atom headline", "https://example.org/a")]


def test_malformed_body_raises_in_parse_but_fetch_never_raises():
    try:
        news_feed.parse_feed(MALFORMED, symbol="NVDA", feed="yahoo", now=NOW)
    except ValueError as exc:
        assert "malformed" in str(exc)
    else:
        raise AssertionError("a malformed body must not read as an empty feed")
    errors: dict[str, str] = {}
    items = news_feed.fetch_symbol("NVDA", now=NOW, get=_get({"yahoo": MALFORMED, "google": GOOGLE_RSS}), errors=errors)
    assert "malformed" in errors["yahoo"] and "google" not in errors
    assert items and all(item.url for item in items)


def test_fetch_symbol_merges_both_feeds_one_per_url_and_sends_timeout_and_agent():
    calls: list = []
    items = news_feed.fetch_symbol("NVDA", now=NOW, get=_get({"yahoo": YAHOO_RSS, "google": GOOGLE_RSS}, calls))
    urls = [item.url for item in items]
    assert len(urls) == len(set(urls))
    dup = next(item for item in items if item.url.endswith("nvidia-amd-rally.html"))
    assert dup.feed == "yahoo", "the primary feed wins a shared URL"
    assert [c[0].split("/")[2] for c in calls] == ["feeds.finance.yahoo.com", "news.google.com"]
    assert all(c[1]["timeout"] == 10 and "TradingBotV3" in c[1]["headers"]["User-Agent"] for c in calls)
    assert "q=NVDA+stock" in calls[1][0] and "s=NVDA" in calls[0][0]


def test_http_error_and_exception_are_reasons_not_raises():
    errors: dict[str, str] = {}
    items = news_feed.fetch_symbol("NVDA", now=NOW, get=_get({"yahoo": _Resp("", 429), "google": TimeoutError("slow")}),
                                   errors=errors)
    assert items == [] and "429" in errors["yahoo"] and "TimeoutError" in errors["google"]


def test_a_bad_ticker_is_never_fetched():
    fetcher = news_feed.NewsFetcher(get=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
    assert fetcher.fetch("NV DA; drop", now=NOW).reason == "not a ticker"
    assert news_feed.feed_urls("") == []


def test_fetcher_min_interval_cycle_cap_and_last_error():
    calls: list = []
    fetcher = news_feed.NewsFetcher(get=_get({"yahoo": YAHOO_RSS, "google": _Resp("", 503)}, calls), max_per_cycle=2)
    first = fetcher.fetch("NVDA", now=NOW)
    assert first.fetched and first.headlines and fetcher.last_error == {"yahoo": "", "google": "ValueError: HTTP 503"}
    again = fetcher.fetch("NVDA", now=NOW + timedelta(minutes=29))
    assert not again.fetched and "too soon" in again.reason and again.headlines == []
    assert fetcher.fetch("AMD", now=NOW + timedelta(minutes=1)).fetched
    capped = fetcher.fetch("TSLA", now=NOW + timedelta(minutes=2))
    assert "cycle cap" in capped.reason
    assert len(calls) == 4, "2 symbols x 2 feeds; the capped and too-soon asks never hit the network"
    later = fetcher.fetch("TSLA", now=NOW + timedelta(minutes=31))
    assert later.fetched, "a new 30-min cycle opens the cap again"


def test_fetcher_default_cap_is_40_per_cycle():
    fetcher = news_feed.NewsFetcher(get=_get({"yahoo": ATOM, "google": ATOM}))
    fetched = [fetcher.fetch(f"S{i}", now=NOW + timedelta(seconds=i)).fetched for i in range(45)]
    assert fetched.count(True) == 40 and fetcher.cycle_count == 40


def test_seed_restores_the_interval_after_a_restart():
    fetcher = news_feed.NewsFetcher(get=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
    fetcher.seed("NVDA", NOW - timedelta(minutes=5))
    assert not fetcher.due("NVDA", NOW) and "too soon" in fetcher.fetch("NVDA", now=NOW).reason


def test_module_is_pure():
    source = (SCRIPTS_DIR / "news_feed.py").read_text(encoding="utf-8")
    assert "PySide6" not in source and "import ui" not in source and "from ui" not in source
    assert "market_prep" not in source
