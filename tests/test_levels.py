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

    def _touch_break_frame(self, periods: int = 60, *, start: str = "2024-01-01") -> pd.DataFrame:
        dates = pd.bdate_range(start, periods=periods)
        rows = []
        for idx, dt_value in enumerate(dates):
            if idx % 6 == 0:
                low, high, close = 99.7, 101.5, 101.2  # break up through 100
            elif idx % 6 == 3:
                low, high, close = 99.5, 100.3, 100.0  # respect 100
            else:
                low, high, close = 103.0, 105.0, 104.0
            rows.append(
                {"datetime": dt_value, "open": 104.0, "high": high, "low": low, "close": close, "volume": 1000.0}
            )
        return pd.DataFrame(rows)

    def test_accumulate_matches_full_recompute_over_overlapping_windows(self):
        frame = self._touch_break_frame()
        level = {
            "kind": "hv_horizontal",
            "price": 100.0,
            "bucket": "green",
            "first_seen": frame.loc[0, "datetime"].date().isoformat(),
        }
        full = levels.recompute_touch_stats([dict(level)], frame, atr20=4.0)[0]
        acc = [dict(level)]
        for start, end in ((0, 25), (18, 45), (40, 60)):
            acc = levels.accumulate_touch_stats(acc, frame.iloc[start:end].reset_index(drop=True), atr20=4.0)
        accumulated = acc[0]
        self.assertEqual(accumulated["touch_count"], full["touch_count"])
        self.assertEqual(accumulated["respect_count"], full["respect_count"])
        self.assertEqual(accumulated["break_count"], full["break_count"])
        self.assertTrue(accumulated["stats_through"])

    def test_accumulate_does_not_double_count_on_replay(self):
        frame = self._touch_break_frame()
        level = {
            "kind": "hv_horizontal",
            "price": 100.0,
            "bucket": "green",
            "first_seen": frame.loc[0, "datetime"].date().isoformat(),
        }
        once = levels.accumulate_touch_stats([dict(level)], frame, atr20=4.0)
        twice = levels.accumulate_touch_stats(once, frame, atr20=4.0)
        self.assertEqual(once[0]["touch_count"], twice[0]["touch_count"])
        self.assertEqual(once[0]["break_count"], twice[0]["break_count"])
        # A short trailing window must never shrink the accumulated totals.
        short = levels.accumulate_touch_stats(twice, frame.iloc[-2:].reset_index(drop=True), atr20=4.0)
        self.assertGreaterEqual(short[0]["touch_count"], once[0]["touch_count"])

    def test_accumulate_migrates_legacy_level_without_double_counting(self):
        frame = self._touch_break_frame(periods=30)
        level = {
            "kind": "hv_horizontal",
            "price": 100.0,
            "bucket": "green",
            "first_seen": frame.loc[0, "datetime"].date().isoformat(),
        }
        legacy_level = levels.recompute_touch_stats([dict(level)], frame, atr20=4.0)[0]
        self.assertNotIn("stats_through", legacy_level)
        migrated = levels.accumulate_touch_stats([dict(legacy_level)], frame, atr20=4.0)[0]
        fresh = levels.accumulate_touch_stats([dict(level)], frame, atr20=4.0)[0]
        self.assertEqual(migrated["touch_count"], fresh["touch_count"])

    def test_level_conviction_grows_with_respect_and_discounts_breaks(self):
        fresh = {"kind": "hv_horizontal", "bucket": "green", "respect_count": 0, "break_count": 0}
        respected = {"kind": "hv_horizontal", "bucket": "green", "respect_count": 8, "break_count": 0}
        flaky = {"kind": "hv_horizontal", "bucket": "green", "respect_count": 4, "break_count": 4}
        red = {"kind": "hv_horizontal", "bucket": "red", "respect_count": 0, "break_count": 0}
        self.assertGreater(levels.level_conviction(respected), levels.level_conviction(fresh))
        self.assertGreater(levels.level_conviction(fresh), levels.level_conviction(red))
        self.assertLess(levels.level_conviction(flaky), levels.level_conviction(respected))
        # A green level fully respected should clear the "full penalty" conviction.
        self.assertGreaterEqual(levels.level_conviction(respected), 1.6)

    def test_merge_into_store_preserves_cumulative_stats(self):
        frame = self._touch_break_frame()
        level = {
            "kind": "hv_horizontal",
            "price": 100.0,
            "bucket": "green",
            "first_seen": frame.loc[0, "datetime"].date().isoformat(),
        }
        accumulated = levels.accumulate_touch_stats([dict(level)], frame, atr20=4.0)[0]
        store = levels.default_level_store("T")
        store["levels"] = [dict(accumulated)]
        reconfirm = {
            "kind": "hv_horizontal",
            "price": 100.02,
            "bucket": "green",
            "relvol": 3.0,
            "first_seen": frame.loc[0, "datetime"].date().isoformat(),
            "touch_count": 0,
            "respect_count": 0,
            "break_count": 0,
        }
        merged = levels.merge_into_store(store, [reconfirm], symbol="T", atr20=4.0)
        matched = next(item for item in merged["levels"] if abs(float(item["price"]) - 100.0) < 0.1)
        self.assertEqual(matched["touch_count"], accumulated["touch_count"])
        self.assertEqual(matched["stats_through"], accumulated["stats_through"])

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

    def test_hv_level_block_penalty_covers_entry_path_not_just_touches(self):
        from master_avwap_lib.legacy import _hv_level_context_for_entry

        wall = {
            "kind": "hv_horizontal",
            "price": 103.0,
            "bucket": "green",
            "strength": 1.0,
            "respect_count": 8,
            "break_count": 0,
        }
        store = {"levels": [wall]}

        # Long entering 0.3 ATR under a respected wall: blocked and penalized,
        # even though it is far outside the tight 0.05-ATR touch tolerance.
        ctx = _hv_level_context_for_entry(
            store, side="LONG", entry_price=100.0, atr20=10.0, scoring_enabled=True
        )
        self.assertEqual(ctx["hv_level_blocking_count"], 1)
        self.assertLess(ctx["hv_level_score_delta"], 0.0)
        self.assertIn("blocking", ctx["hv_level_note"])

        # Closer to the wall = a larger penalty.
        point_blank = _hv_level_context_for_entry(
            store, side="LONG", entry_price=102.8, atr20=10.0, scoring_enabled=True
        )
        self.assertLess(point_blank["hv_level_score_delta"], ctx["hv_level_score_delta"])

        # The same wall BELOW a long entry is support, not a blocker.
        above = _hv_level_context_for_entry(
            store, side="LONG", entry_price=106.0, atr20=10.0, scoring_enabled=True
        )
        self.assertEqual(above["hv_level_blocking_count"], 0)
        self.assertEqual(above["hv_level_score_delta"], 0.0)

        # Outside the entry-path window: no block.
        far = _hv_level_context_for_entry(
            store, side="LONG", entry_price=103.0 - 10.0 * 0.7, atr20=10.0, scoring_enabled=True
        )
        self.assertEqual(far["hv_level_blocking_count"], 0)

        # Shorts invert: the wall below the entry blocks the short.
        short_ctx = _hv_level_context_for_entry(
            store, side="SHORT", entry_price=106.0, atr20=10.0, scoring_enabled=True
        )
        self.assertEqual(short_ctx["hv_level_blocking_count"], 1)
        self.assertLess(short_ctx["hv_level_score_delta"], 0.0)

    def test_hv_level_block_penalty_respects_flag_and_break_today(self):
        from master_avwap_lib.legacy import _hv_level_context_for_entry

        wall = {
            "kind": "hv_horizontal",
            "price": 103.0,
            "bucket": "green",
            "strength": 1.0,
            "respect_count": 8,
            "break_count": 0,
            "last_break": "2026-07-06",
        }
        store = {"levels": [wall]}

        # Flag off: context fields populate but the score is untouched.
        ctx = _hv_level_context_for_entry(
            store, side="LONG", entry_price=100.0, atr20=10.0, scoring_enabled=False
        )
        self.assertEqual(ctx["hv_level_blocking_count"], 1)
        self.assertEqual(ctx["hv_level_score_delta"], 0.0)

        # A wall broken on the latest bar is a breakout, not a fade: no penalty
        # even when the level sits outside the tight nearby tolerance.
        broke = _hv_level_context_for_entry(
            store,
            side="LONG",
            entry_price=100.0,
            atr20=10.0,
            last_trade_date="2026-07-06",
            scoring_enabled=True,
        )
        self.assertTrue(broke["hv_level_break_today"])
        self.assertEqual(broke["hv_level_score_delta"], 0.0)


    def test_find_relative_pivots_detects_strict_swings_and_ignores_plateaus(self):
        prices = [100.0] * 40
        prices[19] = 95.0
        prices[20] = 90.0  # strict relative low
        prices[21] = 95.0
        dates = pd.bdate_range("2024-01-01", periods=40)
        rows = [
            {"datetime": d, "open": p, "high": p, "low": p, "close": p, "volume": 1.0}
            for d, p in zip(dates, prices)
        ]
        frame = pd.DataFrame(rows)

        pivots = levels.find_relative_pivots(frame, lookback=8, atr20=5.0)

        # Only the genuine swing low is a pivot; the flat 100 plateau is not.
        self.assertEqual([(p["kind"], p["bar_index"]) for p in pivots], [("low", 20)])
        self.assertEqual(pivots[0]["price"], 90.0)

    def test_find_relative_pivots_requires_minimum_bars(self):
        frame = _daily_frame(days=10)
        self.assertEqual(levels.find_relative_pivots(frame, lookback=8), [])

    def test_level_store_path_avoids_windows_reserved_device_names(self):
        levels_dir = Path("levels")
        self.assertEqual(levels.level_store_path(levels_dir, "CON").name, "CON_.json")
        self.assertEqual(levels.level_store_path(levels_dir, "NUL").name, "NUL_.json")
        self.assertEqual(levels.level_store_path(levels_dir, "COM1").name, "COM1_.json")
        self.assertEqual(levels.level_store_path(levels_dir, "BRK.B").name, "BRKB.json")
        self.assertEqual(levels.level_store_path(levels_dir, "AAPL").name, "AAPL.json")


if __name__ == "__main__":
    unittest.main()
