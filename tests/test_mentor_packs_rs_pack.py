"""P17 rs_pack: the industry / sector board as leaders, laggards and the book's groups."""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import attach  # noqa: E402
from mentor_packs import registry, rs_pack  # noqa: E402


def test_golden_industry_pack():
    pack = rs_pack.fixture()
    by_id = {row["id"]: row["text"] for row in pack.rows}
    assert list(by_id) == ["rs:asof", "rs:lead:1", "rs:lead:2", "rs:lag:1", "rs:lag:2",
                           "rs:you:insurance", "rs:you:semiconductors"]
    assert by_id["rs:asof"] == ("Industry board built 2026-09-30 16:39 ET (20 min ago), "
                                "5 industry groups ranked by RS (1 = strongest)")
    assert by_id["rs:lead:1"] == ("lead industry #1: Cybersecurity, 11 members, 1d +2.1%, 5d +0.0%; "
                                  "yours: CRWD (liked)")
    assert by_id["rs:lead:2"] == ("lead industry #2: Semiconductors, 40 members, 1d +1.2%, 5d +3.4%; "
                                  "yours: NVDA (book LONG), AMD (Focus)")
    assert by_id["rs:lag:1"].startswith("lag industry #5: Autos") and by_id["rs:lag:1"].endswith("TSLA (Focus)")
    assert by_id["rs:lag:2"].endswith("ALL (book SHORT)")
    assert by_id["rs:you:insurance"] == "Your Insurance (ALL SHORT): rank 4 of 5, 1d -0.8%, 5d -2.0%"
    assert len(pack.ids) == len(set(pack.ids))
    assert datetime.fromisoformat(pack.rows[0]["at_utc"]).tzinfo is not None


def test_sector_level_maps_yahoo_sector_names():
    pack = rs_pack.build(level="sector", top=1, now=rs_pack.FIXTURE_NOW, sources=rs_pack.fixture_sources())
    by_id = {row["id"]: row["text"] for row in pack.rows}
    assert by_id["rs:lead:1"].startswith("lead sector #1: Technology (XLK)")
    assert "NVDA (book LONG)" in by_id["rs:lead:1"]
    # "Financial Services" (classification) is the board's "Financials".
    assert by_id["rs:lag:1"].startswith("lag sector #3: Financials") and "ALL (book SHORT)" in by_id["rs:lag:1"]


def test_stale_board_says_its_age():
    src = rs_pack.fixture_sources(built="2026-09-25T13:00:00")
    pack = rs_pack.build(now=rs_pack.FIXTURE_NOW, sources=src)
    assert pack.ids == ("rs:none",)
    assert "stale" in pack.rows[0]["text"] and "days ago" in pack.rows[0]["text"]


def test_previous_session_board_is_not_stale():
    src = rs_pack.fixture_sources(built="2026-09-29T13:00:00")
    assert rs_pack.build(now=rs_pack.FIXTURE_NOW, sources=src).ids[0] == "rs:asof"


def test_missing_board_is_none():
    src = replace(rs_pack.fixture_sources(), snapshot=lambda: None)
    pack = rs_pack.build(now=rs_pack.FIXTURE_NOW, sources=src)
    assert pack.ids == ("rs:none",) and "unknown" in pack.rows[0]["text"]


def test_registered_and_attached():
    assert "rs_pack" in registry.names()
    now = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)
    for question in ("where's the relative strength right now in industries", "what's leading today",
                     "am I positioned with the strong groups", "which sectors are lagging"):
        names = [request.name for request in attach.plan_attachments(question, {}, now)]
        assert "rs_pack" in names, question
    sector = [r for r in attach.plan_attachments("which sectors are lagging", {}, now) if r.name == "rs_pack"]
    assert sector[0].args["level"] == "sector"


def test_runtime_on_a_copy_of_the_live_board(tmp_path):
    from industry_context import _read_csv_rows

    live = Path(r"C:\TradingBotData\output\industry_indexes.csv")
    if not live.exists():
        pytest.skip("no live board on this machine")
    copy = tmp_path / "industry_indexes.csv"
    shutil.copy(live, copy)
    rows = _read_csv_rows(copy)
    src = replace(rs_pack.fixture_sources(), industry_rows=lambda: rows)
    started = time.perf_counter()
    pack = rs_pack.build(top=5, now=rs_pack.FIXTURE_NOW, sources=src)
    assert time.perf_counter() - started < 0.5
    assert "rs:lead:5" in pack.ids and "rs:lag:5" in pack.ids
