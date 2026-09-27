"""Long leaders golden: `long_setups.build_rows` on real-shaped 2026-09-25 bars.

The fixture (`tests/build_long_setups_golden_fixture.py`) freezes the output on the scan's
top Long leaders names, 60 seeded other names and SPY from the machine cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import long_setups as ls  # noqa: E402


def _bars(rows):
    return [{"date": row[0], "open": row[1], "high": row[2], "low": row[3], "close": row[4], "volume": row[5]}
            for row in rows]


def build(raw):
    return ls.build_rows(
        bars_by_symbol={symbol: _bars(rows) for symbol, rows in raw["bars"].items()},
        spy_bars=_bars(raw["spy"]), feature_rows=raw["feature_rows"],
        earnings_by_symbol=raw["earnings"], atr_by_symbol=raw["atr"], as_of=raw["as_of"])


def _golden():
    from conftest import load_fixture_contract

    return load_fixture_contract("long_setups_golden_v1")


def test_the_long_leaders_output_matches_the_golden():
    golden = _golden()
    assert build(golden["raw"]) == golden["expected"]


def test_the_golden_is_not_empty():
    rows = _golden()["expected"]["rows"]
    assert len(rows) >= 15
    assert sum(1 for row in rows if row["promoted"]) == ls.PROMOTE_MAX
    assert {row["strength_filter"] for row in rows} >= {"yes", "no"}
