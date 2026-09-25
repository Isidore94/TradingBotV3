"""P2-8 8a - Day Review shows the morning's three-axis read and its grade."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

AXES = {
    "line": "Market read (from 2026-09-23 close): SPY trending up · breadth weak (35% > SMA20)",
    "grades": [
        {"axis": "spy", "state": "trending_up", "lean": "up", "verdict": "wrong", "move_atr": -0.9},
        {"axis": "breadth", "state": "weak", "lean": "down", "verdict": "right", "move_atr": -0.9},
    ],
    "summary": {"right": 1, "wrong": 1, "flat": 0, "pending": 0, "no_call": 0, "unmeasured": 0},
}


def test_the_glance_carries_the_market_axes():
    import day_report_card

    glance = day_report_card.glance({"market_axes": AXES})
    assert glance["market_axes"]["summary"]["right"] == 1


def test_the_market_tile_says_right_and_wrong_with_each_axis_in_the_tooltip():
    from ui.widgets.day_glance_strip import TILES, tile_texts

    assert "market_axes" in dict(TILES)
    value, tip = tile_texts({"trades": 0, "market_axes": AXES})["market_axes"]
    assert value == "1 / 1"
    assert "breadth weak" in tip
    assert "spy: wrong" in tip and "breadth: right" in tip


def test_no_read_is_not_measured_and_an_open_day_is_pending():
    from ui.widgets.day_glance_strip import tile_texts

    assert tile_texts({"trades": 0})["market_axes"][0] == "not measured"
    pending = dict(AXES, summary={"right": 0, "wrong": 0, "flat": 0, "pending": 2, "no_call": 0, "unmeasured": 0})
    assert tile_texts({"trades": 0, "market_axes": pending})["market_axes"][0] == "pending"


def test_the_service_puts_the_axes_in_the_payload(monkeypatch):
    import market_axes
    from ui.services import day_review_service

    assert "market_axes" in day_review_service.empty_payload("2026-09-24")
    monkeypatch.setattr(market_axes, "day_axes", lambda session, now: dict(AXES, session=session))
    got = day_review_service.DayReviewService._market_axes("2026-09-24", None)
    assert got["session"] == "2026-09-24"


def test_a_failing_axes_read_costs_only_the_axes(monkeypatch):
    import market_axes
    from ui.services import day_review_service

    def boom(*_a, **_k):
        raise OSError("no tape")

    monkeypatch.setattr(market_axes, "day_axes", boom)
    assert day_review_service.DayReviewService._market_axes("2026-09-24", None) == {}
