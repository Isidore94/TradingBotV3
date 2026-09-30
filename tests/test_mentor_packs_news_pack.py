"""Trade Mentor news pack (P7): stored headlines only, each with its URL, newest first, read-only."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs import news_pack, registry  # noqa: E402
from news_feed import Headline  # noqa: E402

NOW = news_pack.FIXTURE_NOW


def _rows(pack):
    return {row["id"]: row for row in pack.rows}


def test_registered_as_a_tool_with_symbol_and_days():
    assert "news_pack" in registry.names()
    params = registry.modules()["news_pack"].SCHEMA["function"]["parameters"]
    assert params["required"] == ["symbol"] and set(params["properties"]) == {"symbol", "days"}


def test_fixture_golden():
    assert news_pack.fixture().as_text() == (
        "## news_pack\n"
        "[news:NVDA:asof] News for NVDA, last 3 days (Yahoo Finance / Google News RSS; last fetched Tue 09-29 06:30 PT)\n"
        '[news:NVDA:12] Headline "Broadcom reports tomorrow; chips mixed" (finance.yahoo.com, Tue 09-29 05:10 PT) '
        "https://finance.yahoo.com/news/avgo\n"
        '[news:NVDA:11] Headline "Nvidia adds $150 billion buyback" (nytimes.com, Mon 09-28 06:49 PT) '
        "https://www.nytimes.com/nvda-buyback"
    )


def test_every_headline_row_carries_its_url_source_and_a_tz_aware_time():
    for row in news_pack.fixture().rows:
        if row["kind"] != "news":
            continue
        assert row["url"].startswith("https://") and row["url"] in row["text"]
        assert row["source"] in row["text"] and " PT)" in row["text"]
        assert datetime.fromisoformat(row["published_utc"]).tzinfo is not None


def test_a_stored_row_without_a_url_never_renders():
    reader = news_pack.list_reader([
        {"id": 1, "symbol": "NVDA", "title": "No url", "url": "", "published_utc": "2026-09-29T10:00:00+00:00"},
        {"id": 2, "symbol": "NVDA", "title": "Bad url", "url": "ftp://x/y", "published_utc": "2026-09-29T10:00:00+00:00"},
    ])
    pack = news_pack.build("NVDA", now=NOW, reader=reader, stamps=lambda s: "2026-09-29T13:00:00+00:00")
    assert [row["kind"] for row in pack.rows] == ["asof", "news_empty"]
    assert _rows(pack)["news:NVDA:none"]["text"] == "Headlines: none stored in the last 3 days"


def test_never_fetched_says_so_and_max_eight_newest_first():
    many = [{"id": i, "symbol": "AMD", "title": f"S{i}", "url": f"https://e.com/{i}", "source": "e.com",
             "published_utc": (NOW - timedelta(hours=i)).isoformat()} for i in range(1, 12)]
    pack = news_pack.build("amd", 3, now=NOW, reader=news_pack.list_reader(many), stamps=lambda s: None)
    heads = [row["id"] for row in pack.rows if row["kind"] == "news"]
    assert heads == [f"news:AMD:{i}" for i in range(1, 9)]
    assert "never fetched" in _rows(pack)["news:AMD:asof"]["text"]
    empty = news_pack.build("ZZZ", now=NOW, reader=news_pack.list_reader([]), stamps=lambda s: None)
    assert _rows(empty)["news:ZZZ:none"]["text"] == "Headlines: not fetched yet in the last 3 days"


def test_days_window_and_clamp():
    reader = news_pack.fixture_reader()
    week = news_pack.build("NVDA", 14, now=NOW, reader=reader, stamps=news_pack.fixture_stamps)
    assert "news:NVDA:5" in week.ids and "news:NVDA:5" not in news_pack.fixture().ids
    assert news_pack.clamp_days(99) == 14 and news_pack.clamp_days("x") == 3 and news_pack.clamp_days(0) == 1


def test_a_bad_ticker_or_a_broken_reader():
    assert news_pack.build("NV DA; drop", now=NOW).ids == ()

    def broken(*_):
        raise OSError("gone")

    pack = news_pack.build("NVDA", now=NOW, reader=broken, stamps=lambda s: None)
    assert "unknown" in _rows(pack)["news:NVDA:none"]["text"]


def test_store_round_trip_through_the_read_only_reader(tmp_path):
    db = tmp_path / "mentor_chat.sqlite3"
    store = MentorChatStore(db)
    items = [
        Headline("Nvidia rallies", "https://e.com/a", "e.com", "2026-09-29T12:00:00+00:00", "NVDA", "yahoo"),
        Headline("Same URL again", "https://e.com/a", "e.com", "2026-09-29T12:05:00+00:00", "NVDA", "google"),
        Headline("No url", "", "e.com", "2026-09-29T12:00:00+00:00", "NVDA", "yahoo"),
        Headline("", "https://e.com/b", "e.com", "2026-09-29T12:00:00+00:00", "NVDA", "yahoo"),
        Headline("Undated", "https://e.com/c", "e.com", "", "NVDA", "yahoo"),
        Headline("AMD story", "https://e.com/a", "e.com", "2026-09-29T11:00:00+00:00", "AMD", "yahoo"),
    ]
    assert store.put_headlines(items, "2026-09-29T13:00:00+00:00") == 3
    assert store.put_headlines(items[:1], "2026-09-29T13:30:00+00:00") == 0, "one row per symbol + URL"
    assert store.set_news_fetched("NVDA", "2026-09-29T13:00:00+00:00")
    stored = store.headlines("NVDA", since="2026-09-28T00:00:00+00:00")
    assert [row["title"] for row in stored] == ["Undated", "Nvidia rallies"]
    pack = news_pack.build("NVDA", now=NOW, reader=news_pack.db_reader(db), stamps=news_pack.db_stamp_reader(db))
    ids = [row["id"] for row in pack.rows if row["kind"] == "news"]
    assert ids == [f"news:NVDA:{row['id']}" for row in stored]
    assert "time unknown" in pack.rows[1]["text"] and "last fetched" in pack.rows[0]["text"]
    assert store.news_fetch_stamps() == {"NVDA": "2026-09-29T13:00:00+00:00"}


def test_the_reader_never_creates_the_store(tmp_path):
    missing = tmp_path / "none.sqlite3"
    assert news_pack.db_reader(missing)("NVDA", "", 8) == []
    assert news_pack.db_stamp_reader(missing)("NVDA") is None
    assert not missing.exists()


def test_the_reader_on_a_store_from_before_p7(tmp_path):
    import sqlite3

    db = tmp_path / "old.sqlite3"
    sqlite3.connect(db).close()
    assert news_pack.db_reader(db)("NVDA", "", 8) == []


@pytest.mark.parametrize("name", ["news_pack.py"])
def test_the_pack_is_pure(name):
    source = (SCRIPTS_DIR / "mentor_packs" / name).read_text(encoding="utf-8")
    assert "PySide6" not in source and "requests" not in source and "mentor_app" not in source
    assert "news_feed" not in source, "the pack reads the store; it never fetches"
