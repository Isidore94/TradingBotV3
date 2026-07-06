import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib.expected_r import (  # noqa: E402
    DEFAULT_EXPECTED_R_CONFIG,
    _isotonic_non_decreasing,
    blend_expected_r,
    calibrate_prior_anchors,
    compute_expected_r,
    compute_proven_quality_score,
    freshness_factor,
    quality_points_to_prior_r,
    resolved_config,
    shrinkage_weight,
    wilson_lower_bound,
)


class ProvenQualityScoreTests(unittest.TestCase):
    def test_wilson_bound_penalizes_small_samples(self):
        # 2-for-2 must not look like a confident 100% win rate.
        tiny = wilson_lower_bound(1.0, 2)
        seasoned = wilson_lower_bound(0.60, 40)
        self.assertLess(tiny, 0.60)
        self.assertGreater(seasoned, wilson_lower_bound(0.60, 8))
        self.assertIsNone(wilson_lower_bound(None, 10))
        self.assertIsNone(wilson_lower_bound(0.5, 0))

    def test_proven_good_beats_unknown_beats_proven_loser(self):
        good = compute_proven_quality_score(
            static_points=120, win_rate=0.60, profit_factor=2.0, closed_samples=25
        )
        unknown = compute_proven_quality_score(static_points=260)
        loser = compute_proven_quality_score(
            static_points=545, win_rate=0.33, profit_factor=0.6, closed_samples=12
        )
        self.assertGreater(good["score"], unknown["score"])
        self.assertGreater(unknown["score"], loser["score"])
        self.assertTrue(good["proven"])
        self.assertFalse(unknown["proven"])

    def test_structure_is_only_a_tiebreaker(self):
        # A 500-point signal stack adds at most the structure cap.
        modest = compute_proven_quality_score(
            static_points=100, win_rate=0.55, profit_factor=1.5, closed_samples=20
        )
        stacked = compute_proven_quality_score(
            static_points=500, win_rate=0.55, profit_factor=1.5, closed_samples=20
        )
        self.assertLessEqual(stacked["score"] - modest["score"], 40.0)

    def test_freshness_decays_evidence_not_structure(self):
        fresh = compute_proven_quality_score(
            static_points=100, win_rate=0.6, profit_factor=2.0, closed_samples=25, freshness=1.0
        )
        stale = compute_proven_quality_score(
            static_points=100, win_rate=0.6, profit_factor=2.0, closed_samples=25, freshness=0.5
        )
        self.assertLess(stale["score"], fresh["score"])
        self.assertEqual(stale["structure"], fresh["structure"])


class QualityPointsToPriorRTests(unittest.TestCase):
    def test_anchors_are_hit_exactly(self):
        self.assertAlmostEqual(quality_points_to_prior_r(60), -0.20)
        self.assertAlmostEqual(quality_points_to_prior_r(100), 0.30)
        self.assertAlmostEqual(quality_points_to_prior_r(140), 0.70)
        self.assertAlmostEqual(quality_points_to_prior_r(180), 1.05)

    def test_interpolates_between_anchors(self):
        # halfway between 100 (0.30) and 140 (0.70) -> 0.50
        self.assertAlmostEqual(quality_points_to_prior_r(120), 0.50)

    def test_clamps_beyond_ends(self):
        self.assertAlmostEqual(quality_points_to_prior_r(10), -0.20)
        self.assertAlmostEqual(quality_points_to_prior_r(500), 1.05)

    def test_monotonic_non_decreasing(self):
        prev = None
        for pts in range(40, 220, 5):
            value = quality_points_to_prior_r(pts)
            if prev is not None:
                self.assertGreaterEqual(value, prev - 1e-9)
            prev = value

    def test_none_points_treated_as_zero(self):
        self.assertAlmostEqual(quality_points_to_prior_r(None), -0.20)


class ShrinkageWeightTests(unittest.TestCase):
    def test_zero_samples_zero_weight(self):
        self.assertEqual(shrinkage_weight(0), 0.0)
        self.assertEqual(shrinkage_weight(None), 0.0)

    def test_tracker_leads_reaches_half_weight_at_k(self):
        # default tracker_leads k = 3 -> n=3 gives weight 0.5
        self.assertAlmostEqual(shrinkage_weight(3), 0.5)

    def test_tracker_leads_dominates_with_more_samples(self):
        # tracker should clearly lead with a modest sample
        self.assertGreater(shrinkage_weight(9), 0.7)

    def test_mode_changes_aggressiveness(self):
        cfg_balanced = {"mode": "balanced"}
        cfg_conservative = {"mode": "conservative"}
        n = 8
        w_leads = shrinkage_weight(n)
        w_balanced = shrinkage_weight(n, cfg_balanced)
        w_conservative = shrinkage_weight(n, cfg_conservative)
        self.assertGreater(w_leads, w_balanced)
        self.assertGreater(w_balanced, w_conservative)

    def test_capped_by_max_blend_weight(self):
        cap = DEFAULT_EXPECTED_R_CONFIG["max_blend_weight"]
        self.assertLessEqual(shrinkage_weight(10_000), cap + 1e-9)


class FreshnessFactorTests(unittest.TestCase):
    def test_fresh_signals_full_weight(self):
        self.assertEqual(freshness_factor(0), 1.0)
        self.assertEqual(freshness_factor(2), 1.0)

    def test_decays_after_full_window(self):
        # day 5 -> 1 - 0.08*(5-2) = 0.76
        self.assertAlmostEqual(freshness_factor(5), 0.76)

    def test_floored_at_min_factor(self):
        self.assertAlmostEqual(freshness_factor(100), 0.40)

    def test_negative_days_treated_as_fresh(self):
        self.assertEqual(freshness_factor(-3), 1.0)


class BlendExpectedRTests(unittest.TestCase):
    def test_no_samples_returns_prior(self):
        result = blend_expected_r(0.30, 1.5, 0)
        self.assertEqual(result["expected_r"], 0.30)
        self.assertEqual(result["blend_weight"], 0.0)
        self.assertIsNone(result["realized_r"])

    def test_realized_none_returns_prior(self):
        result = blend_expected_r(0.30, None, 5)
        self.assertEqual(result["expected_r"], 0.30)

    def test_tracker_leads_pulls_toward_realized(self):
        # prior modest, realized strong, decent sample -> expected well above prior
        result = blend_expected_r(0.30, 1.6, 9)
        self.assertGreater(result["expected_r"], 0.9)
        self.assertLess(result["expected_r"], 1.6)

    def test_negative_realized_drags_expectation_down(self):
        result = blend_expected_r(0.70, -1.0, 9)
        self.assertLess(result["expected_r"], 0.0)

    def test_realized_is_clipped(self):
        clipped = blend_expected_r(0.30, 50.0, 9)
        bounded = blend_expected_r(0.30, 4.0, 9)
        self.assertAlmostEqual(clipped["expected_r"], bounded["expected_r"])
        self.assertEqual(clipped["realized_r"], 4.0)

    def test_small_sample_shrinks_back_to_prior(self):
        # a single +4R fluke should not dominate a 0.30 prior
        fluke = blend_expected_r(0.30, 4.0, 1)
        strong = blend_expected_r(0.30, 4.0, 12)
        self.assertLess(fluke["expected_r"], strong["expected_r"])
        self.assertLess(fluke["expected_r"], 1.5)


class ComputeExpectedRTests(unittest.TestCase):
    def test_full_pipeline_fresh(self):
        result = compute_expected_r(
            quality_points=140,
            realized_r=1.2,
            closed_samples=9,
            days_since_signal=0,
        )
        self.assertEqual(result["freshness_factor"], 1.0)
        self.assertAlmostEqual(result["rank_score"], result["expected_r"])
        self.assertGreater(result["expected_r"], 0.7)

    def test_stale_positive_is_penalised_in_rank_score(self):
        fresh = compute_expected_r(
            quality_points=140, realized_r=1.2, closed_samples=9, days_since_signal=0
        )
        stale = compute_expected_r(
            quality_points=140, realized_r=1.2, closed_samples=9, days_since_signal=12
        )
        self.assertEqual(fresh["expected_r"], stale["expected_r"])
        self.assertLess(stale["rank_score"], fresh["rank_score"])

    def test_freshness_never_rewards_negative_expectation(self):
        stale_neg = compute_expected_r(
            quality_points=70, realized_r=-1.2, closed_samples=9, days_since_signal=30
        )
        # rank_score must not be pulled toward zero (improved) by staleness
        self.assertEqual(stale_neg["rank_score"], stale_neg["expected_r"])
        self.assertLess(stale_neg["rank_score"], 0.0)

    def test_hot_lower_quality_can_outrank_cold_higher_quality(self):
        # The whole point: a lower static-quality setup that is *working now*
        # should be able to outrank a higher static-quality setup that is cold.
        hot_b_setup = compute_expected_r(
            quality_points=110, realized_r=1.8, closed_samples=10, days_since_signal=0
        )
        cold_a_setup = compute_expected_r(
            quality_points=150, realized_r=-0.3, closed_samples=10, days_since_signal=0
        )
        self.assertGreater(hot_b_setup["rank_score"], cold_a_setup["rank_score"])


class ResolvedConfigTests(unittest.TestCase):
    def test_partial_override_merges_nested(self):
        cfg = resolved_config({"freshness": {"min_factor": 0.1}})
        self.assertEqual(cfg["freshness"]["min_factor"], 0.1)
        # untouched nested defaults remain
        self.assertEqual(cfg["freshness"]["full_days"], 2)
        self.assertIn("tracker_leads", cfg["shrinkage_k"])

    def test_does_not_mutate_default(self):
        cfg = resolved_config({"mode": "balanced"})
        cfg["prior_anchors"].append([999, 9])
        self.assertNotIn([999, 9], DEFAULT_EXPECTED_R_CONFIG["prior_anchors"])


class IsotonicNonDecreasingTests(unittest.TestCase):
    def test_already_monotone_unchanged(self):
        out = _isotonic_non_decreasing([0.1, 0.3, 0.7], [5, 5, 5])
        self.assertEqual([round(v, 3) for v in out], [0.1, 0.3, 0.7])

    def test_violation_is_pooled(self):
        # middle value dips below its neighbour -> pooled to the weighted mean
        out = _isotonic_non_decreasing([0.2, 0.1, 0.9], [1, 1, 1])
        self.assertLessEqual(out[0], out[1])
        self.assertLessEqual(out[1], out[2])
        self.assertAlmostEqual(out[0], out[1])  # 0.2 and 0.1 pooled to 0.15
        self.assertAlmostEqual(out[0], 0.15)

    def test_weighting_respected_when_pooling(self):
        out = _isotonic_non_decreasing([1.0, 0.0], [3, 1])
        # pooled mean = (1.0*3 + 0.0*1)/4 = 0.75 for both
        self.assertAlmostEqual(out[0], 0.75)
        self.assertAlmostEqual(out[1], 0.75)


class CalibratePriorAnchorsTests(unittest.TestCase):
    def test_insufficient_samples_keeps_default(self):
        result = calibrate_prior_anchors([(100, 0.5)] * 5)
        self.assertTrue(result["used_default"])
        self.assertEqual(result["anchors"], [list(p) for p in DEFAULT_EXPECTED_R_CONFIG["prior_anchors"]])

    def test_fitted_anchors_are_monotonic(self):
        # higher points -> higher realized R, with noise
        import random

        random.seed(7)
        samples = []
        for _ in range(60):
            pts = random.uniform(60, 180)
            r = (pts - 100) / 80.0 + random.uniform(-0.4, 0.4)
            samples.append((pts, r))
        result = calibrate_prior_anchors(samples)
        self.assertFalse(result["used_default"])
        anchor_rs = [r for _, r in result["anchors"]]
        self.assertEqual(anchor_rs, sorted(anchor_rs))
        self.assertGreaterEqual(len(result["anchors"]), 2)

    def test_calibration_reflects_observed_edge_direction(self):
        # A bucket of high-scoring setups that actually lost money should pull
        # the top anchor's R below the optimistic default.
        import random

        random.seed(11)
        samples = []
        for _ in range(80):
            pts = random.uniform(150, 190)
            samples.append((pts, random.uniform(-1.5, -0.5)))  # high score, negative R
        for _ in range(80):
            pts = random.uniform(60, 90)
            samples.append((pts, random.uniform(-0.3, 0.1)))
        result = calibrate_prior_anchors(samples)
        self.assertFalse(result["used_default"])
        top_anchor_r = result["anchors"][-1][1]
        # default top anchor is +1.05R; observed top bucket lost money, so the
        # calibrated top anchor must be well below the default.
        self.assertLess(top_anchor_r, 0.5)

    def test_bins_shrink_toward_default(self):
        # A wildly positive raw observation should be pulled toward the default
        # rather than taken at face value (confidence shrinkage).
        samples = [(100, 2.0)] * 32  # all in one points bucket, +2R raw
        result = calibrate_prior_anchors(samples)
        self.assertFalse(result["used_default"])
        default_at_100 = quality_points_to_prior_r(100)  # 0.30
        for _points, anchor_r in result["anchors"]:
            self.assertGreater(anchor_r, default_at_100)  # data pulls it up
            self.assertLess(anchor_r, 2.0)  # but not all the way to the raw observation


if __name__ == "__main__":
    unittest.main()
