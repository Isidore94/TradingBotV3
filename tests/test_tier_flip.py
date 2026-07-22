"""Golden behavior for the MASTER_AVWAP_D1_TIER_FLIP pure module.

Locks the feature contract before any live wiring (plan.md Milestone 3):
transition-only arming, the small-move ATR radius with its fallback chain,
still-armed gating, strict close-through detection (wick touches and
overnight gaps never fire), the side-aware bounce_type map, and the exact
alert message shape.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bounce_bot_lib import tier_flip


def make_entry(**overrides):
    entry = {
        "symbol": "HOMB",
        "side": "LONG",
        "active_current_scan": True,
        "priority_bucket": "",
        "priority_score": 62.0,
        "setup_family": "post_earnings_52w_break",
        "watchlist_run_date": "2026-07-22",
        "last_close": 30.40,
        "atr20": 1.00,
        "trigger_levels": [
            {
                "trigger_id": "second_dev_break:UPPER_2:30.6500",
                "side": "LONG",
                "action": "break_above",
                "event_type": "second_dev_break",
                "label": "UPPER_2",
                "alert_label": "2nd-dev break",
                "level": 30.65,
                "reason": "A/S upgrade target: clear UPPER_2 resistance.",
                "source": "a_s_upgrade_target",
                "target_tier": "A/S",
                "upgrade_only": True,
            }
        ],
    }
    entry.update(overrides)
    return entry


class BuildTierFlipWatchTests(unittest.TestCase):
    def test_non_sa_symbol_with_close_level_arms(self):
        watch = tier_flip.build_tier_flip_watch_entry(make_entry())
        self.assertIsNotNone(watch)
        self.assertEqual(watch["symbol"], "HOMB")
        self.assertEqual(watch["current_bucket"], "")
        self.assertEqual(watch["atr_source"], "atr20")
        self.assertEqual(len(watch["conditions"]), 1)
        self.assertAlmostEqual(watch["conditions"][0]["distance_atr"], 0.25)

    def test_already_sa_buckets_never_arm(self):
        for bucket in ("favorite_setup", "near_favorite_zone"):
            self.assertIsNone(
                tier_flip.build_tier_flip_watch_entry(make_entry(priority_bucket=bucket)),
                bucket,
            )

    def test_tracked_non_sa_bucket_still_arms(self):
        watch = tier_flip.build_tier_flip_watch_entry(
            make_entry(priority_bucket="stdev_retest_tracking")
        )
        self.assertIsNotNone(watch)
        self.assertEqual(watch["current_bucket"], "stdev_retest_tracking")

    def test_far_level_stays_dormant(self):
        entry = make_entry()
        entry["trigger_levels"][0]["level"] = 30.40 + 0.36  # 0.36 ATR away
        self.assertIsNone(tier_flip.build_tier_flip_watch_entry(entry))

    def test_level_already_passed_is_not_armed(self):
        entry = make_entry(last_close=30.70)  # close already above the level
        self.assertIsNone(tier_flip.build_tier_flip_watch_entry(entry))

    def test_inactive_scan_never_arms(self):
        self.assertIsNone(
            tier_flip.build_tier_flip_watch_entry(make_entry(active_current_scan=False))
        )

    def test_atr_fallback_chain(self):
        # entry atr20 null (the live store today) -> caller-supplied bot ATR
        watch = tier_flip.build_tier_flip_watch_entry(make_entry(atr20=None), atr=1.0)
        self.assertEqual(watch["atr_source"], "bot_atr")
        # no ATR anywhere -> price-fraction pseudo-ATR keeps the symbol armed
        # only for genuinely tiny moves (0.25/0.608 ATR ~ 0.41 > 0.35 radius)
        self.assertIsNone(tier_flip.build_tier_flip_watch_entry(make_entry(atr20=None)))
        near = make_entry(atr20=None)
        near["trigger_levels"][0]["level"] = 30.55
        watch = tier_flip.build_tier_flip_watch_entry(near)
        self.assertEqual(watch["atr_source"], "price_fraction")

    def test_short_side_break_below(self):
        entry = make_entry(
            side="SHORT",
            last_close=30.40,
            trigger_levels=[
                {
                    "trigger_id": "first_dev_break:LOWER_1:30.2000",
                    "action": "break_below",
                    "event_type": "first_dev_break",
                    "label": "LOWER_1",
                    "alert_label": "1st-dev break",
                    "level": 30.20,
                    "reason": "A/S upgrade target: break LOWER_1.",
                }
            ],
        )
        watch = tier_flip.build_tier_flip_watch_entry(entry)
        self.assertIsNotNone(watch)
        self.assertEqual(watch["side"], "SHORT")

    def test_non_break_actions_skipped(self):
        entry = make_entry()
        entry["trigger_levels"][0]["action"] = "bounce_up"
        self.assertIsNone(tier_flip.build_tier_flip_watch_entry(entry))

    def test_conditions_sorted_nearest_first(self):
        entry = make_entry(
            trigger_levels=[
                {"action": "break_above", "event_type": "second_dev_break", "level": 30.70, "trigger_id": "b"},
                {"action": "break_above", "event_type": "first_dev_break", "level": 30.50, "trigger_id": "a"},
            ]
        )
        watch = tier_flip.build_tier_flip_watch_entry(entry)
        self.assertEqual([c["level"] for c in watch["conditions"]], [30.50, 30.70])

    def test_map_level_wrapper_uses_bot_atr(self):
        watch = tier_flip.build_tier_flip_watch(
            {"homb": make_entry(atr20=None)}, atr_by_symbol={"HOMB": 1.0}
        )
        self.assertIn("HOMB", watch)
        self.assertEqual(watch["HOMB"]["atr_source"], "bot_atr")


class DetectTierFlipTriggersTests(unittest.TestCase):
    def setUp(self):
        self.watch = tier_flip.build_tier_flip_watch_entry(make_entry())

    def test_close_through_fires(self):
        bars = [{"close": 30.55}, {"close": 30.72}]
        fired = tier_flip.detect_tier_flip_triggers(self.watch, bars)
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0]["label"], "UPPER_2")

    def test_wick_touch_does_not_fire(self):
        bars = [{"close": 30.55, "high": 30.60}, {"close": 30.60, "high": 30.80}]
        self.assertEqual(tier_flip.detect_tier_flip_triggers(self.watch, bars), [])

    def test_already_through_does_not_refire(self):
        # Both completed closes beyond the level: the cross happened earlier
        # (or gapped overnight) - the alert belongs to the bar of the move.
        bars = [{"close": 30.70}, {"close": 30.80}]
        self.assertEqual(tier_flip.detect_tier_flip_triggers(self.watch, bars), [])

    def test_break_below_mirror(self):
        watch = tier_flip.build_tier_flip_watch_entry(
            make_entry(
                side="SHORT",
                trigger_levels=[
                    {
                        "action": "break_below",
                        "event_type": "first_dev_break",
                        "label": "LOWER_1",
                        "level": 30.20,
                        "trigger_id": "x",
                    }
                ],
            )
        )
        self.assertEqual(
            len(tier_flip.detect_tier_flip_triggers(watch, [{"close": 30.30}, {"close": 30.10}])), 1
        )
        self.assertEqual(
            tier_flip.detect_tier_flip_triggers(watch, [{"close": 30.10}, {"close": 30.05}]), []
        )

    def test_needs_two_completed_bars(self):
        self.assertEqual(tier_flip.detect_tier_flip_triggers(self.watch, [{"close": 30.72}]), [])


class BounceTypeMapTests(unittest.TestCase):
    def test_side_aware_dev_band_mapping(self):
        self.assertEqual(
            tier_flip.bounce_types_for_flip_event("first_dev_break", "LONG"),
            ["dynamic_vwap_upper_band"],
        )
        self.assertEqual(
            tier_flip.bounce_types_for_flip_event("first_dev_break", "SHORT"),
            ["dynamic_vwap_lower_band"],
        )

    def test_known_event_types_map_or_fall_back_empty(self):
        # Every event type the D1 trigger builder can emit (the sort-order
        # table in master_avwap_lib) either maps to a real M5 segment or
        # deliberately contributes none - never a KeyError, never a fabricated
        # segment.
        known_event_types = [
            "avwape_reclaim",
            "avwape_breakdown",
            "post_earnings_52w_break",
            "first_dev_break",
            "second_dev_break",
            "mid_earnings_ema15_retest_watch",
            "mid_earnings_first_dev_retest_watch",
            "avwape_retest_watch",
            "ema15_retest_watch",
            "first_dev_retest_watch",
            "previous_day_high_break",
            "previous_day_low_break",
            "trendline_break_watch",
            "sma_obstacle_break",
            "previous_anchor_obstacle_break",
        ]
        for event_type in known_event_types:
            for side in ("LONG", "SHORT"):
                result = tier_flip.bounce_types_for_flip_event(event_type, side)
                self.assertIsInstance(result, list, event_type)

    def test_unmapped_and_blank(self):
        self.assertEqual(tier_flip.bounce_types_for_flip_event("trendline_break_watch", "LONG"), [])
        self.assertEqual(tier_flip.bounce_types_for_flip_event("", "LONG"), [])


class ContextTierGateTests(unittest.TestCase):
    def test_min_b_default(self):
        self.assertTrue(tier_flip.context_tier_passes("S"))
        self.assertTrue(tier_flip.context_tier_passes("A"))
        self.assertTrue(tier_flip.context_tier_passes("B"))
        self.assertFalse(tier_flip.context_tier_passes("C"))
        self.assertFalse(tier_flip.context_tier_passes("D"))
        self.assertFalse(tier_flip.context_tier_passes(""))

    def test_raised_floor(self):
        self.assertFalse(tier_flip.context_tier_passes("B", minimum="A"))
        self.assertTrue(tier_flip.context_tier_passes("S", minimum="A"))


class FormatTierFlipMessageTests(unittest.TestCase):
    def test_golden_message(self):
        message = tier_flip.format_tier_flip_message(
            {
                "symbol": "HOMB",
                "direction": "long",
                "alert_label": "2nd-dev break",
                "level": 30.65,
                "current_price": 30.72,
                "bar_time": "10:35",
                "distance_atr": 0.25,
                "scan_close": 30.40,
                "priority_score": 62.0,
                "current_bucket": "",
                "reason": "A/S upgrade target: clear UPPER_2 resistance.",
                "context_tier": "B",
                "context_note": "dynamic_vwap_upper_band long entry +0.21R (n=41)",
                "rvol": 1.42,
            }
        )
        self.assertEqual(
            message,
            "MASTER_AVWAP_D1_TIER_FLIP: HOMB (long) non-S/A -> A/S predicted "
            "(next scan confirms) 2nd-dev break [@30.65; px=30.72; bar=10:35; "
            "move 0.25 ATR from scan close 30.40; was: bucket none, score 62; "
            "A/S upgrade target: clear UPPER_2 resistance.; "
            "ctx B-tier: dynamic_vwap_upper_band long entry +0.21R (n=41); rvol 1.42]",
        )

    def test_message_never_contains_parseable_tier_token(self):
        # The Alert Center's live-feed tier regex must not match the context
        # chip - the D1 Focus routing owns this alert, not the tier gate.
        import re

        message = tier_flip.format_tier_flip_message({"symbol": "X", "context_tier": "B"})
        self.assertIsNone(re.search(r"\[([SABCD])-TIER\]", message, re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
