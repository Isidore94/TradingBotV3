import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib import levels  # noqa: E402


def _daily_frame(days: int = 60, *, start: str = "2026-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=days)
    rows = []
    for idx, dt_value in enumerate(dates):
        price = 100.0 + idx * 0.1
        rows.append(
            {
                "datetime": dt_value,
                "open": price - 0.2,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 100.0,
            }
        )
    return pd.DataFrame(rows)


class LevelModuleTests(unittest.TestCase):
    def test_compute_relvol_uses_current_bar_in_sma(self):
        frame = pd.DataFrame(
            {
                "datetime": pd.bdate_range("2026-01-01", periods=3),
                "open": [1, 1, 1],
                "high": [1, 1, 1],
                "low": [1, 1, 1],
                "close": [1, 1, 1],
                "volume": [10, 20, 30],
            }
        )

        relvol = levels.compute_relvol(frame, vol_sma=3)

        self.assertTrue(pd.isna(relvol.iloc[1]))
        self.assertAlmostEqual(relvol.iloc[2], 1.5, places=6)

    def test_extract_hv_levels_emits_high_and_low_with_buckets(self):
        frame = _daily_frame()
        frame.loc[49, "volume"] = 250.0
        frame.loc[55, "volume"] = 600.0
        frame.loc[58, "volume"] = 600.0
        earnings_date = frame.loc[54, "datetime"].date().isoformat()

        candidates = levels.extract_hv_levels(frame, atr20=10.0, earnings_dates=[earnings_date])

        self.assertEqual(len(candidates), 6)
        buckets = [candidate["bucket"] for candidate in candidates]
        self.assertEqual(buckets.count("red"), 2)
        self.assertEqual(buckets.count("green"), 4)
        origin_sides = {candidate["origin_side"] for candidate in candidates}
        self.assertEqual(origin_sides, {"high", "low"})
        green = [candidate for candidate in candidates if candidate["bucket"] == "green"]
        earnings_green = [candidate for candidate in green if candidate["earnings_origin"]]
        non_earnings_green = [candidate for candidate in green if candidate["non_earnings_anchor_candidate"]]
        self.assertEqual(len(earnings_green), 2)
        self.assertEqual(len(non_earnings_green), 2)

    def test_cluster_levels_merges_nearby_prices_by_atr_tolerance(self):
        candidates = [
            {"price": 100.0, "bucket": "red", "relvol": 2.2, "first_seen": "2026-01-01", "origin_side": "high"},
            {"price": 100.2, "bucket": "green", "relvol": 4.0, "first_seen": "2026-01-03", "origin_side": "low"},
            {"price": 104.0, "bucket": "red", "relvol": 2.4, "first_seen": "2026-01-02", "origin_side": "high"},
        ]

        clustered = levels.cluster_levels(candidates, atr20=10.0)

        self.assertEqual(len(clustered), 2)
        self.assertEqual(clustered[0]["bucket"], "green")
        self.assertEqual(clustered[0]["first_seen"], "2026-01-01")
        self.assertEqual(clustered[0]["member_count"], 2)
        self.assertGreater(clustered[0]["price"], 100.0)
        self.assertLess(clustered[0]["price"], 100.2)

    def test_compute_span_b_flats_applies_forward_displacement(self):
        dates = pd.bdate_range("2026-01-01", periods=90)
        frame = pd.DataFrame(
            {
                "datetime": dates,
                "open": [100.0] * 90,
                "high": [110.0] * 62 + [130.0] * 28,
                "low": [90.0] * 62 + [100.0] * 28,
                "close": [100.0] * 90,
                "volume": [1000.0] * 90,
            }
        )

        flats = levels.compute_span_b_flats(frame, atr20=10.0)

        first_flat = next(flat for flat in flats if flat["price"] == 100.0)
        self.assertEqual(first_flat["kind"], "cloud_flat")
        self.assertEqual(first_flat["computed_range"], [dates[51].date().isoformat(), dates[61].date().isoformat()])
        self.assertEqual(first_flat["effective_range"], [dates[77].date().isoformat(), dates[87].date().isoformat()])

    def test_levels_near_filters_cloud_lines_by_effective_range(self):
        store = {
            "levels": [
                {
                    "kind": "cloud_flat",
                    "price": 100.1,
                    "bucket": "cloud",
                    "strength": 1.0,
                    "effective_range": ["2026-03-01", "2026-03-31"],
                },
                {"kind": "hv_horizontal", "price": 100.1, "bucket": "green", "strength": 1.0},
            ]
        }

        active = levels.levels_near(store, 100.0, 10.0, kinds={"cloud_flat"}, as_of_date="2026-03-10")
        inactive = levels.levels_near(store, 100.0, 10.0, kinds={"cloud_flat"}, as_of_date="2026-04-10")
        hv_only = levels.levels_near(store, 100.0, 10.0, kinds={"hv_horizontal"}, as_of_date="2026-04-10")

        self.assertEqual(len(active), 1)
        self.assertEqual(inactive, [])
        self.assertEqual(len(hv_only), 1)

    def test_recompute_touch_stats_is_idempotent(self):
        frame = _daily_frame(days=6)
        frame.loc[1, ["high", "low", "close"]] = [100.2, 99.8, 100.1]
        frame.loc[2, ["high", "low", "close"]] = [103.2, 99.9, 103.0]
        frame.loc[3:, "low"] = 102.0
        frame.loc[3:, "high"] = 104.0
        frame.loc[3:, "close"] = 103.0
        level = {
            "kind": "hv_horizontal",
            "price": 100.0,
            "bucket": "green",
            "first_seen": frame.loc[0, "datetime"].date().isoformat(),
        }

        once = levels.recompute_touch_stats([level], frame, atr20=10.0)
        twice = levels.recompute_touch_stats(once, frame, atr20=10.0)

        self.assertEqual(once, twice)
        self.assertEqual(once[0]["touch_count"], 2)
        self.assertEqual(once[0]["respect_count"], 1)
        self.assertEqual(once[0]["break_count"], 1)

    def test_levels_blocking_entry_filters_by_side_and_strength(self):
        store = {
            "levels": [
                {"kind": "hv_horizontal", "price": 100.4, "bucket": "green", "strength": 1.0},
                {"kind": "hv_horizontal", "price": 99.7, "bucket": "green", "strength": 1.0},
                {"kind": "hv_horizontal", "price": 100.2, "bucket": "red", "strength": 0.35},
            ]
        }

        long_blockers = levels.levels_blocking_entry(store, "LONG", 100.0, 10.0)
        short_blockers = levels.levels_blocking_entry(store, "SHORT", 100.0, 10.0)

        self.assertEqual([item["price"] for item in long_blockers], [100.4])
        self.assertEqual([item["price"] for item in short_blockers], [99.7])


if __name__ == "__main__":
    unittest.main()
