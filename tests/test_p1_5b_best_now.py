"""P1-5 5b: the "Best right now" ranker (its desk box was removed 2026-09-28)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import best_now  # noqa: E402


def _alert(symbol, side="LONG", *, grade="B", r=0.5, status="open", entry=10.0, stop=9.5, at="09:40"):
    return {
        "symbol": symbol,
        "side": side,
        "grade": grade,
        "r": r,
        "status": status,
        "entry": entry,
        "stop": stop,
        "received_at": f"2026-09-24 {at}",
    }


def _dip(symbol, score, *, last=20.0, lod=19.2):
    return {"symbol": symbol, "dip_score": score, "last": last, "lod": lod, "since_start_pct": 0.8}


# --- the pure ranker --------------------------------------------------------


def test_d1_with_m5_outranks_m5_alone_which_outranks_dip_strong():
    entries = best_now.rank_best_now(
        [_alert("AAA", grade="A", r=1.2), _alert("BBB", grade="C", r=0.1)],
        [_dip("CCC", 3.0)],
        {("BBB", "LONG"): {"grade": "B", "family": "retest", "claimed": False}},
    )
    assert [e.symbol for e in entries] == ["BBB", "AAA", "CCC"]
    assert [e.tier for e in entries] == [best_now.TIER_D1_M5, best_now.TIER_M5, best_now.TIER_DIP]
    assert entries[0].why.startswith("D1 B + M5 C")
    assert (entries[0].entry, entries[0].stop) == (10.0, 9.5)
    assert (entries[2].entry, entries[2].stop) == (20.0, 19.2)
    assert entries[2].why.startswith("Dip-strong +3.0")


def test_a_d1_name_without_an_m5_alert_is_not_listed():
    entries = best_now.rank_best_now([], [], {("ZZZ", "LONG"): {"grade": "A"}})
    assert entries == []


def test_inside_a_tier_grade_then_live_r_then_time():
    entries = best_now.rank_best_now(
        [
            _alert("LOWR", grade="A", r=0.2, at="09:35"),
            _alert("HIGHR", grade="A", r=1.5, at="09:50"),
            _alert("BGRADE", grade="B", r=3.0, at="09:31"),
            _alert("NODATA", grade="A", r=None, status="unknown", at="09:30"),
        ]
    )
    assert [e.symbol for e in entries] == ["HIGHR", "LOWR", "NODATA", "BGRADE"]


def test_a_stopped_alert_is_left_out_and_a_dip_name_with_an_alert_merges():
    entries = best_now.rank_best_now(
        [_alert("STOP", status="stopped", r=-1.0), _alert("BOTH", r=0.4)],
        [_dip("BOTH", 2.0), _dip("STOP", 1.0)],
    )
    assert [e.symbol for e in entries] == ["BOTH", "STOP"]
    assert entries[0].tier == best_now.TIER_M5 and "dip-strong" in entries[0].why
    assert entries[1].tier == best_now.TIER_DIP


def test_the_limit_and_the_row_text():
    entries = best_now.rank_best_now([_alert(f"S{i}") for i in range(9)], limit=6)
    assert len(entries) == 6
    text = best_now.entry_text(entries[0])
    assert text == "S0 L  M5 B +0.5R\n e 10.00 s 9.50"


def test_diff_rows_names_only_the_changed_rows():
    assert best_now.diff_rows(["a", "b", "c"], ["a", "b", "c"]) == []
    assert best_now.diff_rows(["a", "b", "c"], ["a", "x", "c"]) == [1]
    assert best_now.diff_rows(["a", "b"], ["a", "b", "c"]) == [2]
    assert best_now.diff_rows(["a", "b", "c"], ["a"]) == [1, 2]


def test_dip_strong_rows_reads_the_board_or_nothing():
    assert best_now.dip_strong_rows({"dip": {"long": [_dip("X", 1.0)], "short": []}})[0]["symbol"] == "X"
    assert best_now.dip_strong_rows({}) == []
    assert best_now.dip_strong_rows(None) == []


# --- the box is gone ---------------------------------------------------------


def test_the_desk_has_no_best_right_now_box():
    """Trader 2026-09-28: the box listed weak names; it is off the desk."""
    source = (ROOT / "scripts" / "ui" / "panels" / "trading_desk.py").read_text(encoding="utf-8")
    assert "BestNowStrip" not in source and "best_now_strip" not in source
    assert not (ROOT / "scripts" / "ui" / "widgets" / "best_now_strip.py").exists()
