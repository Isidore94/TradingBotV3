"""M5 alerts that sit on a D1 swing setup (trader, 2026-09-23). Pure helper."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import swing_context  # noqa: E402


@dataclass
class _Row:
    symbol: str
    side: str
    bucket: str = "favorite_setup"
    raw: dict[str, Any] = field(default_factory=dict)


def _grades(mapping):
    return lambda row: mapping.get(row.symbol)


def test_match_is_by_symbol_and_side():
    rows = [_Row("NVDA", "LONG", raw={"setup_family": "avwap_bounce"}), _Row("AMD", "SHORT")]
    ctx = swing_context.build_swing_context(rows, _grades({"NVDA": {"grade": "A"}}), set())
    assert swing_context.context_for(ctx, "nvda", "LONG") == {
        "grade": "A",
        "family": "avwap_bounce",
        "claimed": False,
    }
    # Only a SHORT setup on AMD: a LONG alert gets nothing.
    assert swing_context.context_for(ctx, "AMD", "LONG") is None
    assert swing_context.suffix(swing_context.context_for(ctx, "AMD", "LONG")) == ""
    assert swing_context.context_for(ctx, "AMD", "SHORT") is not None


def test_missing_grade_shows_new_never_an_invented_grade():
    rows = [_Row("NVDA", "LONG")]
    before_load = swing_context.build_swing_context(rows, lambda _row: None)
    ctx = before_load[("NVDA", "LONG")]
    assert ctx["grade"] is None
    assert swing_context.suffix(ctx) == "· D1 New"
    junk = swing_context.build_swing_context(rows, lambda _row: {"grade": "Z"})
    assert swing_context.suffix(junk[("NVDA", "LONG")]) == "· D1 New"


def test_claimed_star_from_keys_and_from_the_merged_row():
    rows = [
        _Row("NVDA", "LONG"),
        _Row(
            "TSLA",
            "SHORT",
            bucket="claimed_like",
            raw={"claimed_setup_id": "d1_wick", "bucket_keys": ["claimed_like"]},
        ),
    ]
    grades = _grades({"NVDA": {"grade": "A"}, "TSLA": {"grade": "New"}})
    ctx = swing_context.build_swing_context(rows, grades, {("nvda", "long")})
    assert swing_context.suffix(ctx[("NVDA", "LONG")]) == "· D1 A ★"
    assert swing_context.suffix(ctx[("TSLA", "SHORT")]) == "· D1 New ★"
    assert swing_context.tooltip_line(ctx[("TSLA", "SHORT")]) == (
        "D1 setup: d1_wick, grade New, claimed"
    )
    assert swing_context.claimed_keys_from_rows(rows) == {("TSLA", "SHORT")}


def test_two_rows_on_one_name_keep_the_best_grade_and_any_claim():
    rows = [
        _Row("NVDA", "LONG", raw={"setup_family": "weak"}),
        _Row("NVDA", "LONG", raw={"setup_family": "strong"}),
        _Row("NVDA", "LONG", bucket="claimed_like", raw={"claimed_setup_id": "x"}),
    ]
    cells = {"weak": {"grade": "C"}, "strong": {"grade": "A"}}
    ctx = swing_context.build_swing_context(
        rows, lambda row: cells.get(row.raw.get("setup_family")) or {"grade": "New"}
    )
    assert ctx[("NVDA", "LONG")] == {"grade": "A", "family": "strong", "claimed": True}


def test_sort_key_claimed_then_grade_then_no_context():
    keys = {
        "claimed_c": swing_context.sort_key({"grade": "C", "claimed": True}),
        "a": swing_context.sort_key({"grade": "A", "claimed": False}),
        "new": swing_context.sort_key({"grade": None, "claimed": False}),
        "d": swing_context.sort_key({"grade": "D", "claimed": False}),
        "none": swing_context.sort_key(None),
    }
    assert sorted(keys, key=keys.get) == ["claimed_c", "a", "new", "d", "none"]
