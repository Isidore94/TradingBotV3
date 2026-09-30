"""Mentor app P7: news for the coach. /news cards, the 30-min refresh scope and gates, and the rule
that a headline never shows without its URL and the model never cites one it was not given."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import news_feed  # noqa: E402
from mentor_app import assess, commands, gate, news_jobs, settings  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs import gate_pack, news_pack, pick_pack  # noqa: E402
from mentor_packs.citations import CitationRejected  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
TUE_0700 = datetime(2026, 9, 29, 7, 0, tzinfo=PT)

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>{sym} beats on revenue</title><link>https://example.com/{sym}/beat</link>
<pubDate>Tue, 29 Sep 2026 13:00:00 +0000</pubDate></item>
<item><title>{sym} item with no link</title><pubDate>Tue, 29 Sep 2026 13:10:00 +0000</pubDate></item>
</channel></rss>"""


class _Resp:
    status_code = 200

    def __init__(self, body: str) -> None:
        self.content = body.encode("utf-8")


def _fake_get(calls: list):
    def get(url, **kwargs):
        calls.append(url)
        sym = url.split("s=")[1].split("&")[0] if "s=" in url else url.split("q=")[1].split("+")[0]
        return _Resp(RSS.format(sym=sym) if "yahoo" in url else "<rss version='2.0'><channel></channel></rss>")

    return get


@pytest.fixture
def store(tmp_path):
    return MentorChatStore(tmp_path / "mentor_chat.sqlite3")


# ---------------------------------------------------------------- commands and scope
def test_news_command_parses_symbol_and_days():
    assert commands.handle("/news nvda").arg == ("NVDA", 3)
    assert commands.handle("/news NVDA 7").arg == ("NVDA", 7)
    assert commands.handle("/news NVDA 7d").arg == ("NVDA", 7)
    for bad in ("/news", "/news NVDA 30", "/news NV DA x", "/news 123"):
        assert commands.handle(bad).action == "error", bad
    assert "/news SYM" in commands.HELP_TEXT


def test_schedule_is_every_30_min_in_the_session_weekdays_only():
    schedule = news_jobs.NewsSchedule()
    assert not schedule.due(datetime(2026, 9, 29, 5, 59, tzinfo=PT))
    assert schedule.due(TUE_0700)
    schedule.mark(TUE_0700)
    assert not schedule.due(TUE_0700 + timedelta(minutes=29))
    assert schedule.due(TUE_0700 + timedelta(minutes=30))
    assert not schedule.due(datetime(2026, 9, 29, 13, 30, tzinfo=PT)), "13:30 PT closes the window"
    assert not news_jobs.NewsSchedule().due(datetime(2026, 10, 3, 9, 0, tzinfo=PT)), "Saturday"


def test_scope_is_named_then_book_then_24_liked_never_all_focus():
    liked = [f"L{i}" for i in range(30)]
    scope = news_jobs.news_scope(named=["amd", "NVDA"], open_book=["TSLA", "NVDA"], liked=["NVDA", *liked])
    assert scope[:3] == ["AMD", "NVDA", "TSLA"]
    assert scope[3:] == liked[:23], "the liked list is cut at its first 24 (NVDA was one of them)"
    assert "bad ticker!" not in news_jobs.news_scope(named=["bad ticker!"])


# ---------------------------------------------------------------- store + fetch
def test_refresh_stores_headlines_with_urls_and_stamps(store):
    calls: list = []
    fetcher = news_feed.NewsFetcher(get=_fake_get(calls))
    out = news_jobs.refresh_symbol("NVDA", store=store, fetcher=fetcher, now=TUE_0700)
    assert out["fetched"] and out["added"] == 1
    rows = store.headlines("NVDA")
    assert [(r["title"], r["url"]) for r in rows] == [("NVDA beats on revenue", "https://example.com/NVDA/beat")]
    assert store.news_fetched("NVDA") == TUE_0700.astimezone(timezone.utc).isoformat(timespec="seconds")
    again = news_jobs.refresh_symbol("NVDA", store=store, fetcher=fetcher, now=TUE_0700 + timedelta(minutes=5))
    assert not again["fetched"] and len(calls) == 2, "30 min between fetches of one symbol"


def test_the_spacing_survives_a_restart(store):
    news_jobs.refresh_symbol("NVDA", store=store, fetcher=news_feed.NewsFetcher(get=_fake_get([])), now=TUE_0700)
    calls: list = []
    fresh = news_feed.NewsFetcher(get=_fake_get(calls))
    assert news_jobs.seed_fetcher(fresh, store) == 1
    assert not news_jobs.refresh_symbol("NVDA", store=store, fetcher=fresh, now=TUE_0700 + timedelta(minutes=10))["fetched"]
    assert calls == []


def test_news_card_fetches_a_never_fetched_symbol_once_then_reads_the_store(store):
    calls: list = []
    fetcher = news_feed.NewsFetcher(get=_fake_get(calls))
    first = news_jobs.news_card("NVDA", 3, store=store, fetcher=fetcher, now=TUE_0700)
    assert len(calls) == 2
    assert "[NVDA beats on revenue](https://example.com/NVDA/beat)" in first["markdown"]
    assert "no link" not in first["markdown"], "a headline without a URL never shows"
    assert "Tue 09-29 06:00 PT" in first["markdown"]
    second = news_jobs.news_card("NVDA", 3, store=store, fetcher=fetcher, now=TUE_0700 + timedelta(hours=2))
    assert len(calls) == 2, "a fetched symbol's /news reads the store; the 30-min job refreshes it"
    assert "https://example.com/NVDA/beat" in second["markdown"]


def test_card_markdown_never_shows_a_headline_row_without_a_url():
    pack = news_pack.make_pack("news_pack", [
        {"id": "news:X:asof", "kind": "asof", "text": "News for X"},
        {"id": "news:X:1", "kind": "news", "title": "No url", "url": "", "source": "s", "published_utc": ""},
        {"id": "news:X:2", "kind": "news", "title": "Ok", "url": "https://e.com/2", "source": "e.com",
         "published_utc": "2026-09-29T13:00:00+00:00"},
    ])
    text = news_jobs.card_markdown(pack)
    assert "No url" not in text and "[Ok](https://e.com/2)" in text


# ---------------------------------------------------------------- narration: citations and URLs
def _pick_pack(tmp_path):
    return pick_pack.build("NVDA", now=pick_pack.FIXTURE_NOW, paths=pick_pack.write_fixture_world(tmp_path))


def test_a_cited_headline_renders_with_its_url_in_the_pick_card(tmp_path):
    pack = _pick_pack(tmp_path)
    reply = {"summary": {"verdict": "wait", "rule_flags": [], "bullets": [
        {"text": "Broadcom reports tomorrow.", "evidence_refs": ["pick:NVDA:news:12", "pick:NVDA:peer:AVGO"]}]}}
    card = assess.assess(pack, symbol="NVDA", pack_hash="h", model="m", endpoint="http://x", request=lambda **_: reply)
    text = assess.card_markdown(card)
    assert card.narrated
    assert "[Broadcom reports tomorrow; chips mixed](https://finance.yahoo.com/news/avgo)" in text


def test_a_bullet_citing_a_headline_not_in_the_pack_is_rejected_and_never_shown(tmp_path):
    pack = _pick_pack(tmp_path)
    reply = {"summary": {"verdict": "pass", "rule_flags": [], "bullets": [
        {"text": "Nvidia is being sued.", "evidence_refs": ["pick:NVDA:news:999"]},
        {"text": "The cell is fine.", "evidence_refs": ["pick:NVDA:cell"]}]}}
    with pytest.raises(CitationRejected):
        assess.check_reply(reply["summary"], pack)
    card = assess.assess(pack, symbol="NVDA", pack_hash="h", model="m", endpoint="http://x", request=lambda **_: reply)
    assert not card.narrated and "rejected" in card.error and card.bullets == []
    assert "sued" not in assess.card_markdown(card)


def test_a_gate_card_shows_a_cited_headline_with_its_url(tmp_path):
    pack = gate_pack.build("SHORT", "NVDA", 400, 3.2, 3.05, now=gate_pack.FIXTURE_NOW,
                           sources=gate_pack.fixture_sources(tmp_path))
    reply = {"summary": {"verdict": "wait", "rule_flags": [], "bullets": [
        {"text": "A buyback headline.", "evidence_refs": ["gate:NVDA:pick:NVDA:news:11"]}]}}
    card = gate.narrate(pack, symbol="NVDA", pack_hash="h", model="m", endpoint="http://x", request=lambda **_: reply)
    text = gate.card_markdown(card, gate.CheckRequest("SHORT", "NVDA", 400, 3.2, 3.05), pack)
    assert "[Nvidia adds $150 billion buyback](https://www.nytimes.com/nvda-buyback)" in text
    rejected = gate.narrate(pack, symbol="NVDA", pack_hash="h", model="m", endpoint="http://x", request=lambda **_: {
        "summary": {"verdict": "go", "rule_flags": [], "bullets": [{"text": "x", "evidence_refs": ["news:NVDA:11"]}]}})
    assert not rejected.narrated and "rejected" in rejected.error, "the bare news id is not the gate's id"


def test_the_prompts_say_a_headline_is_title_and_link_only():
    for task in (assess.TASK, gate.TASK):
        assert "never mention news that is not a row here" in task


# ---------------------------------------------------------------- the window
@pytest.fixture
def win(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    clock = {"now": TUE_0700}
    calls: list = []
    desk = {"free": False}
    model_calls: list = []
    window = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"),
        stream_post=lambda *a, **k: model_calls.append(a) or [], post=lambda *a, **k: model_calls.append(a) or {},
        now=lambda: clock["now"], mentor_enabled=False,
        liked_source=lambda: [(f"L{i}", "LONG") for i in range(30)],
        news_fetcher=news_feed.NewsFetcher(get=_fake_get(calls)),
        news_open_symbols=lambda: ["TSLA"],
        desk_probe=lambda: desk["free"],
        assess_request=lambda **k: model_calls.append(k) or {},
    )
    window.clock, window.calls, window.desk, window.model_calls = clock, calls, desk, model_calls
    yield window
    window.shutdown()
    window.deleteLater()


def _drain(window):
    while window.queue.run_one():
        pass


def test_news_command_with_the_brain_down_is_a_card_with_urls_and_no_model(win):
    assert not win._brain_ok
    win.send("/news NVDA")
    _drain(win)
    text = win.transcript.toPlainText()
    assert "NVDA beats on revenue" in text and "no link" not in text
    html = win.transcript.toHtml()
    assert "https://example.com/NVDA/beat" in html
    assert win.model_calls == [] and "NVDA" in win._news_named


def test_news_command_while_ai_is_paused_still_answers(win):
    import ai_pause

    ai_pause.pause_for("2h", win.clock["now"])
    try:
        win.check_ai_pause()
        win.send("/news AMD 2")
        _drain(win)
        assert "AMD beats on revenue" in win.transcript.toPlainText() and win.model_calls == []
    finally:
        ai_pause.resume()


def test_the_refresh_fetches_the_scope_capped_quietly_and_never_the_inbox(win):
    import ai_pause

    win.send("/news AMD")
    _drain(win)
    before_blocks, before_inbox = list(win._blocks), win.inbox.badge()
    ai_pause.pause_for("2h", win.clock["now"])
    try:
        win.check_ai_pause()
        win.maybe_refresh_news()
        assert win.queue.pending() == ["news_plan"]
        _drain(win)
    finally:
        ai_pause.resume()
    fetched = {url.split("s=")[1].split("&")[0] for url in win.calls if "yahoo" in url}
    assert "TSLA" in fetched and "L0" in fetched and "L23" in fetched and "L24" not in fetched
    assert len(fetched) == 26, "AMD (named, fetched by /news) + TSLA + 24 liked chips; never all of Focus"
    assert win._news_fetcher.cycle_count <= 40
    assert win._blocks == before_blocks, "a refresh never moves the transcript"
    assert win.inbox.badge() == before_inbox and not any(i.kind == "news" for i in win.inbox.items())
    assert not any(name.startswith("pick") for name in win.queue.ran), "a new headline never narrates a card"
    assert win.model_calls == []


def test_the_refresh_waits_for_its_window_and_skips_a_closed_desk(win):
    win.clock["now"] = datetime(2026, 9, 29, 5, 30, tzinfo=PT)
    win.maybe_refresh_news()
    assert win.queue.pending() == []
    win.clock["now"] = TUE_0700
    win.desk["free"] = True
    win.maybe_refresh_news()
    _drain(win)
    assert win.calls == [], "the desk is closed: no fetch"


def test_a_refresh_at_the_cap_fetches_at_most_40(win):
    win._liked_source = lambda: [(f"L{i}", "LONG") for i in range(30)]
    win._news_named.extend(f"N{i}" for i in range(30))
    win.maybe_refresh_news()
    _drain(win)
    fetched = {url.split("s=")[1].split("&")[0] for url in win.calls if "yahoo" in url}
    assert len(fetched) == 40 and win._news_fetcher.cycle_count == 40


def test_news_jobs_never_touch_the_qt_thread_or_the_inbox_module():
    source = (SCRIPTS_DIR / "mentor_app" / "news_jobs.py").read_text(encoding="utf-8")
    assert "PySide6" not in source and "post_to_inbox" not in source and "inbox.add" not in source
