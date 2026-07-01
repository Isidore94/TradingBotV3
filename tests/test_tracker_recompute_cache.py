"""The tracker recompute memoization must be strictly behavior-preserving.

`update_setup_tracker_from_scan` replays every tracked setup each scan, and many
setups share a symbol. To avoid rebuilding the (pure) indicator frame and
anchored-VWAP band history once per record, `recompute_tracker_setup_record` now
accepts a shared `indicator_frame` and a per-symbol `band_history_cache`. These
tests pin that the cached path produces records identical to the inline path, and
that a shared cache keyed by anchor date does not cross-contaminate setups with
different anchors.
"""

import copy
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402


def _daily_frame(periods: int = 60) -> pd.DataFrame:
    dates = pd.bdate_range("2026-03-02", periods=periods)
    rows = []
    for i, dt in enumerate(dates):
        base = 100.0 + i * 0.3
        rows.append(
            {
                "datetime": dt,
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.2,
                "volume": 1_000_000 + i * 1_000,
            }
        )
    return pd.DataFrame(rows)


def _setup(df: pd.DataFrame, *, anchor_offset: int = 5, entry_offset: int = 10) -> dict:
    anchor_date = df.iloc[anchor_offset]["datetime"].date().isoformat()
    entry_date = df.iloc[entry_offset]["datetime"].date().isoformat()
    entry_price = float(df.iloc[entry_offset]["close"])
    return {
        "symbol": "TEST",
        "side": "LONG",
        "priority_bucket": "favorite_setup",
        "setup_family": "avwap_retest_followthrough",
        "entry_price": entry_price,
        "entry_trade_date": entry_date,
        "scan_date": entry_date,
        "anchor_date": anchor_date,
        "scenarios": {
            "s1": {
                "tradeable": True,
                "shares": 100,
                "stop_reference_label": "LOWER_1",
                "stop_reference_level": entry_price - 3.0,
                "status": "OPEN",
            }
        },
    }


class RecomputeCacheEquivalenceTests(unittest.TestCase):
    def test_cached_record_matches_inline_record(self):
        df = _daily_frame()
        setup = _setup(df)

        inline = m.recompute_tracker_setup_record(copy.deepcopy(setup), df)
        cached = m.recompute_tracker_setup_record(
            copy.deepcopy(setup),
            df,
            indicator_frame=m.compute_indicator_frame(df),
            band_history_cache={},
        )
        # The band history must actually have been exercised (anchor is in-range).
        self.assertTrue(inline.get("daily_marks"))
        self.assertEqual(inline, cached)

    def test_shared_cache_does_not_leak_across_anchors(self):
        df = _daily_frame()
        setup_a = _setup(df, anchor_offset=5, entry_offset=10)
        setup_b = _setup(df, anchor_offset=20, entry_offset=25)
        self.assertNotEqual(setup_a["anchor_date"], setup_b["anchor_date"])

        # Independent inline baselines.
        inline_a = m.recompute_tracker_setup_record(copy.deepcopy(setup_a), df)
        inline_b = m.recompute_tracker_setup_record(copy.deepcopy(setup_b), df)

        # One shared per-symbol cache + shared indicator frame across both records.
        shared_cache: dict = {}
        indicator_frame = m.compute_indicator_frame(df)
        cached_a = m.recompute_tracker_setup_record(
            copy.deepcopy(setup_a), df, indicator_frame=indicator_frame, band_history_cache=shared_cache
        )
        cached_b = m.recompute_tracker_setup_record(
            copy.deepcopy(setup_b), df, indicator_frame=indicator_frame, band_history_cache=shared_cache
        )

        self.assertEqual(inline_a, cached_a)
        self.assertEqual(inline_b, cached_b)
        # Both anchors got cached under their own keys.
        self.assertIn(setup_a["anchor_date"], shared_cache)
        self.assertIn(setup_b["anchor_date"], shared_cache)

    def test_missing_anchor_is_handled_without_cache(self):
        df = _daily_frame()
        setup = _setup(df)
        setup.pop("anchor_date", None)
        inline = m.recompute_tracker_setup_record(copy.deepcopy(setup), df)
        cached = m.recompute_tracker_setup_record(
            copy.deepcopy(setup), df, indicator_frame=m.compute_indicator_frame(df), band_history_cache={}
        )
        self.assertEqual(inline, cached)


if __name__ == "__main__":
    unittest.main()
