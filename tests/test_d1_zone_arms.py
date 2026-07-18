"""Fixture tests for the pure D1 band-zone arm builder.

Golden-fixture coverage (plan.md Milestone 3) for build_d1_zone_arms before it is
wired into the live M5 alert path: every zone on both sides, the prior-1st-dev
condition, the zone-3 sustained gate, and the still-armed direction gating.
"""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib.d1_zone_arms import (  # noqa: E402
    build_d1_zone_arms,
    detect_zone_arm_triggers,
)


def _bar(high, low, close):
    return {"high": high, "low": low, "close": close}


# A clean symmetric ladder: 1 stdev == 5 points; ATR 5 -> bounce tolerance 1.0.
LADDER = dict(
    avwape=100.0,
    upper_1=105.0,
    upper_2=110.0,
    upper_3=115.0,
    lower_1=95.0,
    lower_2=90.0,
    lower_3=85.0,
    ema15=102.0,
    ema21=101.0,
    stdev=5.0,
    atr=5.0,
)


def _by_label_action(entry):
    return {(arm["label"], arm["action"]): arm for arm in entry["trigger_levels"]}


class D1ZoneArmTests(unittest.TestCase):
    def test_long_zone1_arms_avwape_bounce_and_1st_dev_break(self):
        entry = build_d1_zone_arms(symbol="nvda", close=103.0, **LADDER)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["side"], "LONG")
        self.assertEqual(entry["zone"], 1)
        self.assertEqual(entry["tolerance"], 1.0)
        arms = _by_label_action(entry)
        self.assertIn(("AVWAPE", "bounce_up"), arms)
        self.assertIn(("UPPER_1", "break_above"), arms)
        self.assertEqual(arms[("AVWAPE", "bounce_up")]["level"], 100.0)
        self.assertEqual(arms[("AVWAPE", "bounce_up")]["tolerance"], 1.0)
        self.assertEqual(arms[("UPPER_1", "break_above")]["tolerance"], 0.0)
        # No prior-anchor arm without prev_upper_1.
        self.assertNotIn(("PREV_UPPER_1", "bounce_up"), arms)

    def test_long_zone1_arms_prior_1st_dev_only_when_below_current(self):
        below = build_d1_zone_arms(symbol="NVDA", close=103.0, prev_upper_1=101.0, **LADDER)
        self.assertIn(("PREV_UPPER_1", "bounce_up"), _by_label_action(below))
        # Prior UPPER_1 at/above the current one is not a fresh reclaim -> no arm.
        above = build_d1_zone_arms(symbol="NVDA", close=103.0, prev_upper_1=106.0, **LADDER)
        self.assertNotIn(("PREV_UPPER_1", "bounce_up"), _by_label_action(above))

    def test_long_zone2_arms_2nd_dev_break_and_reclaims(self):
        entry = build_d1_zone_arms(symbol="AMD", close=107.0, **LADDER)
        self.assertEqual(entry["zone"], 2)
        arms = _by_label_action(entry)
        self.assertIn(("UPPER_2", "break_above"), arms)
        self.assertIn(("UPPER_1", "bounce_up"), arms)
        self.assertIn(("EMA_15", "bounce_up"), arms)
        self.assertIn(("EMA_21", "bounce_up"), arms)
        self.assertFalse(any(arm["critical"] for arm in entry["trigger_levels"]))

    def test_zone2_skips_ema_bounce_above_price(self):
        # A 15EMA sitting above the close cannot be a pullback-reclaim level.
        params = dict(LADDER)
        params["ema15"] = 108.0
        entry = build_d1_zone_arms(symbol="AMD", close=107.0, **params)
        arms = _by_label_action(entry)
        self.assertNotIn(("EMA_15", "bounce_up"), arms)
        self.assertIn(("EMA_21", "bounce_up"), arms)

    def test_long_zone3_is_critical_pullbacks_only(self):
        entry = build_d1_zone_arms(symbol="META", close=112.0, sustained_2nd_3rd=True, **LADDER)
        self.assertEqual(entry["zone"], 3)
        arms = _by_label_action(entry)
        self.assertEqual(
            set(arms),
            {("UPPER_1", "bounce_up"), ("EMA_15", "bounce_up"), ("EMA_21", "bounce_up")},
        )
        self.assertTrue(all(arm["critical"] for arm in entry["trigger_levels"]))
        # No breakout arm in zone 3.
        self.assertFalse(any(arm["action"].startswith("break") for arm in entry["trigger_levels"]))

    def test_zone3_requires_sustained_flag(self):
        self.assertIsNone(build_d1_zone_arms(symbol="META", close=112.0, sustained_2nd_3rd=False, **LADDER))

    def test_no_zone_beyond_third_dev_returns_none(self):
        self.assertIsNone(build_d1_zone_arms(symbol="META", close=120.0, **LADDER))

    def test_short_zone1_mirrors_long(self):
        entry = build_d1_zone_arms(symbol="TSLA", close=97.0, prev_lower_1=99.0, **LADDER)
        self.assertEqual(entry["side"], "SHORT")
        self.assertEqual(entry["zone"], 1)
        arms = _by_label_action(entry)
        self.assertIn(("AVWAPE", "bounce_down"), arms)
        self.assertIn(("LOWER_1", "break_below"), arms)
        self.assertIn(("PREV_LOWER_1", "bounce_down"), arms)

    def test_short_zone2_arms(self):
        params = dict(LADDER)
        params["ema15"] = 93.0
        params["ema21"] = 94.0
        entry = build_d1_zone_arms(symbol="TSLA", close=92.0, **params)
        self.assertEqual(entry["zone"], 2)
        arms = _by_label_action(entry)
        self.assertEqual(
            set(arms),
            {
                ("LOWER_2", "break_below"),
                ("LOWER_1", "bounce_down"),
                ("EMA_15", "bounce_down"),
                ("EMA_21", "bounce_down"),
            },
        )

    def test_short_zone3_is_critical_pullbacks_only(self):
        params = dict(LADDER)
        params["ema15"] = 93.0
        params["ema21"] = 94.0
        entry = build_d1_zone_arms(symbol="TSLA", close=87.0, sustained_2nd_3rd=True, **params)
        self.assertEqual(entry["zone"], 3)
        arms = _by_label_action(entry)
        self.assertEqual(
            set(arms),
            {("LOWER_1", "bounce_down"), ("EMA_15", "bounce_down"), ("EMA_21", "bounce_down")},
        )
        self.assertTrue(all(arm["critical"] for arm in entry["trigger_levels"]))

    def test_missing_bands_returns_none(self):
        self.assertIsNone(build_d1_zone_arms(symbol="X", close=None, avwape=100.0))
        self.assertIsNone(build_d1_zone_arms(symbol="", close=100.0, avwape=100.0))


class DetectZoneArmTriggerTests(unittest.TestCase):
    def setUp(self):
        # Long zone 1: bounce_up AVWAPE@100 (tol 1.0), break_above UPPER_1@105.
        self.long_zone1 = build_d1_zone_arms(symbol="NVDA", close=103.0, **LADDER)

    def _labels(self, fired):
        return {(arm["label"], arm["action"]) for arm in fired}

    def test_two_bar_bounce_up_fires_on_reclaim(self):
        bars = [
            _bar(101.0, 100.2, 100.5),  # bar A tags AVWAPE and holds
            _bar(103.0, 100.8, 102.0),  # bar B closes higher and back above
        ]
        fired = detect_zone_arm_triggers(self.long_zone1, bars)
        self.assertIn(("AVWAPE", "bounce_up"), self._labels(fired))
        self.assertNotIn(("UPPER_1", "break_above"), self._labels(fired))

    def test_bounce_up_does_not_fire_without_reclaim(self):
        bars = [
            _bar(101.0, 100.2, 100.5),  # tags
            _bar(100.6, 99.0, 99.5),    # closes below the level, not a reclaim
        ]
        self.assertEqual(detect_zone_arm_triggers(self.long_zone1, bars), [])

    def test_bounce_up_needs_the_tag_bar_first(self):
        bars = [
            _bar(103.5, 103.0, 103.2),  # never came down to AVWAPE
            _bar(104.0, 103.1, 103.8),
        ]
        self.assertEqual(detect_zone_arm_triggers(self.long_zone1, bars), [])

    def test_break_above_fires_on_single_bar_cross(self):
        bars = [
            _bar(104.0, 103.0, 104.0),  # prior close below UPPER_1
            _bar(106.0, 104.0, 105.5),  # reaches through UPPER_1
        ]
        fired = detect_zone_arm_triggers(self.long_zone1, bars)
        self.assertIn(("UPPER_1", "break_above"), self._labels(fired))
        self.assertNotIn(("AVWAPE", "bounce_up"), self._labels(fired))

    def test_two_bar_bounce_down_for_shorts(self):
        params = dict(LADDER)
        params["ema15"] = 93.0
        params["ema21"] = 94.0
        short_zone2 = build_d1_zone_arms(symbol="TSLA", close=92.0, **params)
        bars = [
            _bar(94.8, 93.0, 94.5),  # bar A pushes up to reject LOWER_1@95
            _bar(94.0, 91.0, 92.0),  # bar B closes lower and back below
        ]
        fired = detect_zone_arm_triggers(short_zone2, bars)
        self.assertIn(("LOWER_1", "bounce_down"), self._labels(fired))

    def test_fewer_than_two_completed_bars_never_fires(self):
        self.assertEqual(detect_zone_arm_triggers(self.long_zone1, [_bar(101.0, 100.2, 100.5)]), [])
        self.assertEqual(detect_zone_arm_triggers(self.long_zone1, []), [])


if __name__ == "__main__":
    unittest.main()
