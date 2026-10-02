"""AVWAPE quick test golden: `avwape_quick_test.build_rows` on real 2026-10-01 bars.

The fixture (`tests/build_avwape_quick_test_golden_fixture.py`) freezes the output on every
cached name that fired on 2026-10-01, names that fired in the four sessions before (negatives
today) and seeded controls, from the machine cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import avwape_quick_test as aqt  # noqa: E402


def _bars(rows):
    return [{"date": row[0], "open": row[1], "high": row[2], "low": row[3], "close": row[4], "volume": row[5]}
            for row in rows]


def build(raw):
    return aqt.build_rows(
        bars_by_symbol={symbol: _bars(rows) for symbol, rows in raw["bars"].items()},
        spy_bars=_bars(raw["spy"]), feature_rows=raw["feature_rows"], atr_by_symbol=raw["atr"],
        earnings_dates_by_symbol=raw["earnings_dates"], as_of=raw["as_of"])


def _golden():
    from conftest import load_fixture_contract

    return load_fixture_contract("avwape_quick_test_golden_v1")


def test_the_quick_test_output_matches_the_golden():
    golden = _golden()
    assert build(golden["raw"]) == golden["expected"]


def test_the_golden_holds_real_positives():
    golden = _golden()
    rows = golden["expected"]["rows"]
    assert rows, "the golden must hold at least one real fire"
    assert golden["expected"]["as_of"] == golden["raw"]["as_of"]
    assert all(row["setup"] == aqt.SETUP and row["status"] == aqt.STATUS_TESTING for row in rows)
    # Names that fired only on an earlier session are kept as today's negatives.
    assert len(golden["raw"]["bars"]) > len({row["symbol"] for row in rows})
