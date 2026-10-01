"""P16: the gaps the 100-question live eval found (book scope, hold by outcome, veto aggregates, tape diff, routes)."""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import attach  # noqa: E402

NOW = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)  # Wed 11:00 ET


def _desk():
    """Focus shorts LULU, MKC, GIS; a liked short WULF; the book holds NVDA and DRAM short and QCOM long."""
    rows = [{"kind": "focus", "category": "swing", "side": "short", "names": ["LULU", "MKC", "GIS"]},
            {"kind": "position", "symbol": "NVDA", "direction": "SHORT"},
            {"kind": "position", "symbol": "DRAM", "direction": "SHORT"},
            {"kind": "position", "symbol": "QCOM", "direction": "LONG"}]
    likes = [("WULF", "SHORT")]
    return rows, likes, attach.known_symbols(rows, likes, [])


def _plan(question):
    rows, likes, known = _desk()
    return attach.plan_attachments(question, known, NOW, book=attach.book_symbols(rows), liked=likes)


# ---------------------------------------------------------------- step 1: book-only scope
def test_my_open_shorts_into_earnings_is_the_book_only_never_focus():
    for question in ("which of my open shorts is most at risk into earnings", "my shorts reporting soon?",
                     "any earnings in my positions?"):
        got = _plan(question)
        earn = next(r for r in got if r.name == "earnings_pack")
        assert set(earn.args["symbols"]) <= {"NVDA", "DRAM", "QCOM"}, (question, earn.args)
        assert not {"LULU", "MKC", "GIS", "WULF"} & set(earn.args["symbols"]), question
        assert "book_pack" in [r.name for r in got], question
    shorts = next(r for r in _plan("my shorts reporting soon?") if r.name == "earnings_pack")
    assert shorts.args["symbols"] == ["NVDA", "DRAM"] and shorts.args["book"] == ["NVDA", "DRAM"]


def test_my_focus_shorts_and_watchlist_are_focus_and_likes_not_the_book():
    for question in ("any of my focus shorts reporting this week?", "earnings in my watchlist shorts?"):
        earn = next(r for r in _plan(question) if r.name == "earnings_pack")
        assert earn.args["symbols"] == ["WULF", "LULU", "MKC", "GIS"], (question, earn.args)
        assert earn.args["book"] == [] and earn.args["liked"] == ["WULF"]
        assert earn.args["focus"] == ["LULU", "MKC", "GIS"]


def test_every_earnings_row_says_book_or_focus():
    from mentor_packs import earnings_pack, pick_pack

    world = pick_pack.write_fixture_world(tempfile.mkdtemp())
    pack = earnings_pack.build(["NVDA", "TSLA", "ZZZ"], book=["NVDA"], liked=["ZZZ"], focus=["TSLA"], side="SHORT",
                               now=pick_pack.FIXTURE_NOW, paths=world)
    rows = {row["id"]: row for row in pack.rows}
    assert rows["earn:NVDA"]["origin"] == "book" and rows["earn:NVDA"]["text"].startswith("NVDA (book short):")
    assert rows["earn:TSLA"]["origin"] == "focus" and rows["earn:TSLA"]["text"].startswith("TSLA (Focus short):")
    assert rows["earn:ZZZ"]["origin"] == "liked" and rows["earn:ZZZ"]["text"].startswith("ZZZ (liked short):")
    assert "never positions" in rows["earn:asof"]["text"]


def test_news_over_my_shorts_reads_book_names_with_their_origin():
    picks = [r for r in _plan("any news on my shorts?") if r.name == "pick_pack"]
    assert [(r.args["symbol"], r.args["origin"]) for r in picks] == [("NVDA", "book"), ("DRAM", "book")]
    picks = [r for r in _plan("any news on my focus shorts?") if r.name == "pick_pack"]
    assert [(r.args["symbol"], r.args["origin"]) for r in picks] == [
        ("WULF", "liked"), ("LULU", "focus"), ("MKC", "focus")]


def test_a_pick_pack_for_a_focus_name_says_it_is_not_a_position():
    from mentor_packs import pick_pack

    world = pick_pack.write_fixture_world(tempfile.mkdtemp())
    pack = pick_pack.build("TSLA", origin="focus", now=pick_pack.FIXTURE_NOW, paths=world)
    assert "a Focus name, NOT a position" in pack.rows[0]["text"]
    assert all(row["origin"] == "focus" for row in pack.rows)


def test_do_i_have_any_shorts_on_reads_the_book():
    assert "book_pack" in [r.name for r in _plan("do I have any shorts on")]
