"""End-to-end tests for Expected-R wiring against the live legacy module.

These import the heavy ``master_avwap`` module (pandas / ibapi / yfinance), so
they live apart from the fast pure-math tests in ``test_expected_r.py``.
"""

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap  # noqa: E402


def _row(symbol, side, bucket, family, score, *, recent_delta=0, setup_type_delta=0):
    return {
        "symbol": symbol,
        "side": side,
        "priority_bucket": bucket,
        "setup_family": family,
        "score": score,
        "recent_tracker_score_delta": recent_delta,
        "setup_type_score_delta": setup_type_delta,
    }


def _family_row_for(
    row,
    *,
    closed,
    avg_closed_r,
    avg_total_r,
    tracked=None,
    win_rate=None,
    profit_factor=None,
):
    # Build the recent-family aggregate using the SAME canonical context the
    # real pipeline derives, so the lookup key matches what the ranking uses.
    ctx = master_avwap._tracker_setup_context(row)
    return {
        "side": ctx["side"],
        "priority_bucket": ctx["priority_bucket"],
        "setup_family": ctx["setup_family"],
        "closed_setups": closed,
        "avg_closed_r": avg_closed_r,
        "avg_total_r": avg_total_r,
        "tracked_setups": tracked if tracked is not None else closed,
        "win_rate_closed": win_rate,
        "profit_factor": profit_factor,
    }


class ApplyExpectedRRankingTests(unittest.TestCase):
    def test_signal_age_family_golden_cases(self):
        reference = date(2026, 7, 13)
        cases = [
            ({"setup_family": "mid_earnings_ema15_retest", "mid_earnings_retest_date": "2026-07-09"}, 4, "2026-07-09"),
            ({"setup_family": "sma_breakout_retest", "sma_breakout_breakout_date": "2026-07-05", "sma_breakout_retest_date": "2026-07-10", "sma_breakout_confirmation_date": "2026-07-12"}, 1, "2026-07-12"),
            ({"setup_family": "post_earnings_avwap_bounce", "post_earnings_bounce_date": "2026-07-10"}, 3, "2026-07-10"),
            ({"setup_family": "avwap_breakout", "last_trade_date": "2026-07-13"}, 0, "2026-07-13"),
            ({"setup_family": "general"}, 0, ""),
        ]

        for row, expected_days, expected_date in cases:
            with self.subTest(row=row):
                days, signal_date = master_avwap._expected_r_signal_age(
                    row, reference_date=reference
                )
                self.assertEqual(days, expected_days)
                self.assertEqual(signal_date, expected_date)

    def test_stale_dated_trigger_scores_below_fresh_trigger(self):
        stale = _row("OLD", "LONG", "favorite_setup", "mid_earnings_ema15_retest", 140)
        fresh = _row("NEW", "LONG", "favorite_setup", "mid_earnings_ema15_retest", 140)
        stale["mid_earnings_retest_date"] = "2026-07-03"
        fresh["mid_earnings_retest_date"] = "2026-07-13"
        family_rows = [
            _family_row_for(
                stale,
                closed=20,
                avg_closed_r=1.0,
                avg_total_r=0.9,
                win_rate=0.60,
                profit_factor=2.0,
            )
        ]

        master_avwap.apply_expected_r_ranking(
            [stale, fresh],
            {"symbols": {}},
            {},
            recent_family_rows=family_rows,
            reference_date=date(2026, 7, 13),
        )

        self.assertLess(stale["expected_r_freshness"], fresh["expected_r_freshness"])
        self.assertLess(stale["score"], fresh["score"])
        self.assertEqual(stale["expected_r_signal_date"], "2026-07-03")
        self.assertEqual(stale["expected_r_signal_age_days"], 10)

    def test_attaches_expected_r_and_mirrors_to_state(self):
        row = _row("NVDA", "LONG", "favorite_setup", "post_earnings_52w_break", 150)
        family_rows = [_family_row_for(row, closed=9, avg_closed_r=1.4, avg_total_r=1.2, tracked=12)]
        ai_state = {"symbols": {"NVDA": {}}}
        feature = {"NVDA": {}}

        master_avwap.apply_expected_r_ranking(
            [row], ai_state, feature, recent_family_rows=family_rows
        )

        self.assertIn("expected_r", row)
        self.assertGreater(row["expected_r"], 0.7)
        self.assertEqual(int(row["expected_r_samples"]), 9)
        self.assertGreater(row["expected_r_blend_weight"], 0.5)
        # Mirrored onto ai_state symbol entry and feature row for downstream use.
        self.assertEqual(ai_state["symbols"]["NVDA"]["expected_r"], row["expected_r"])
        self.assertEqual(feature["NVDA"]["expected_r"], row["expected_r"])

    def test_no_tracker_sample_falls_back_to_prior(self):
        row = _row("AAPL", "LONG", "favorite_setup", "post_earnings_52w_break", 150)
        ai_state = {"symbols": {"AAPL": {}}}
        master_avwap.apply_expected_r_ranking([row], ai_state, {}, recent_family_rows=[])
        self.assertEqual(row["expected_r_blend_weight"], 0.0)
        self.assertEqual(row["expected_r"], row["expected_r_prior"])

    def test_quality_points_strip_tracker_deltas(self):
        # Two rows with identical *static* quality but different baked-in tracker
        # deltas must get the same prior (no double-counting).
        a = _row("AAA", "LONG", "favorite_setup", "fam", 150, recent_delta=20, setup_type_delta=10)
        b = _row("BBB", "LONG", "favorite_setup", "fam", 120, recent_delta=0, setup_type_delta=0)
        master_avwap.apply_expected_r_ranking([a, b], {"symbols": {}}, {}, recent_family_rows=[])
        self.assertAlmostEqual(a["expected_r_prior"], b["expected_r_prior"])

    def test_stacked_signals_cannot_outrank_proven_quality(self):
        # The TPG case, evolved: a stacked-signal 545-point setup from a losing
        # family must score BELOW a modest but proven winner and below unknowns.
        loser = _row("TPG", "SHORT", "near_favorite_zone", "mid_earnings_ema15_retest", 545)
        winner = _row("GEN", "LONG", "near_favorite_zone", "avwap_breakout", 140)
        unknown = _row("NEWB", "LONG", "watch", "brand_new_family", 260)
        family_rows = [
            _family_row_for(loser, closed=12, avg_closed_r=-0.6, avg_total_r=-0.5, win_rate=0.33, profit_factor=0.6),
            _family_row_for(winner, closed=20, avg_closed_r=1.1, avg_total_r=1.0, win_rate=0.60, profit_factor=2.0),
        ]

        rows = [loser, winner, unknown]
        master_avwap.apply_expected_r_ranking(rows, {"symbols": {}}, {}, recent_family_rows=family_rows)

        self.assertGreater(winner["score"], unknown["score"])
        self.assertGreater(unknown["score"], loser["score"])
        # Audit trail survives: the raw stack is preserved, points are evidence.
        self.assertEqual(loser["static_score"], 545)
        self.assertEqual(loser["tracker_win_rate"], 0.33)
        self.assertIn("evidence", loser["proven_quality_note"])
        self.assertIn("no closed tracker history", unknown["proven_quality_note"])

    def test_hot_b_setup_outranks_cold_a_setup(self):
        # The headline behaviour: a lower static-quality setup that is working
        # now should rank above a higher static-quality setup that is cold.
        hot = _row("AMD", "LONG", "near_favorite_zone", "mid_earnings_first_dev_retest", 110)
        cold = _row("NVDA", "LONG", "favorite_setup", "post_earnings_52w_break", 150)
        family_rows = [
            _family_row_for(hot, closed=10, avg_closed_r=1.9, avg_total_r=1.7, tracked=12, win_rate=0.7, profit_factor=2.6),
            _family_row_for(cold, closed=10, avg_closed_r=-0.3, avg_total_r=-0.2, tracked=12, win_rate=0.4, profit_factor=0.8),
        ]
        ai_state = {"symbols": {"AMD": {}, "NVDA": {}}}
        rows = [hot, cold]

        master_avwap.apply_expected_r_ranking(
            rows, ai_state, {}, recent_family_rows=family_rows
        )
        ranked = sorted(rows, key=master_avwap._priority_expected_r_sort_key)

        self.assertEqual(ranked[0]["symbol"], "AMD")
        self.assertGreater(hot["expected_r"], cold["expected_r"])
        self.assertGreater(hot["score"], cold["score"])


class ExpectedRCalibrationIntegrationTests(unittest.TestCase):
    def tearDown(self):
        master_avwap._EXPECTED_R_CONFIG_CACHE = None

    def test_load_config_defaults_when_no_file(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "nope.json"
            with patch.object(master_avwap, "EXPECTED_R_CONFIG_FILE", missing):
                master_avwap._EXPECTED_R_CONFIG_CACHE = None
                cfg = master_avwap.load_expected_r_config(force_reload=True)
        self.assertEqual(
            cfg["prior_anchors"],
            master_avwap.DEFAULT_EXPECTED_R_CONFIG["prior_anchors"],
        )

    def test_calibration_round_trip_persists_and_reloads(self):
        # Monotone synthetic history: higher score -> higher realized R.
        samples = [(float(pts), (pts - 100) / 80.0) for pts in range(60, 184, 2)] * 2
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "expected_r.json"
            with patch.object(master_avwap, "EXPECTED_R_CONFIG_FILE", cfg_path), patch.object(
                master_avwap, "build_expected_r_calibration_samples", return_value=samples
            ):
                master_avwap._EXPECTED_R_CONFIG_CACHE = None
                result = master_avwap.calibrate_expected_r_prior_anchors(persist=True)
                self.assertFalse(result["used_default"])
                self.assertTrue(cfg_path.exists())

                reloaded = master_avwap.load_expected_r_config(force_reload=True)
                self.assertEqual(reloaded["prior_anchors"], result["anchors"])
                # anchors stay monotonic non-decreasing in R
                anchor_rs = [r for _, r in reloaded["prior_anchors"]]
                self.assertEqual(anchor_rs, sorted(anchor_rs))

    def test_calibration_keeps_default_when_history_thin(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "expected_r.json"
            with patch.object(master_avwap, "EXPECTED_R_CONFIG_FILE", cfg_path), patch.object(
                master_avwap, "build_expected_r_calibration_samples", return_value=[(100.0, 0.5)] * 4
            ):
                master_avwap._EXPECTED_R_CONFIG_CACHE = None
                result = master_avwap.calibrate_expected_r_prior_anchors(persist=True)
                self.assertTrue(result["used_default"])
                self.assertFalse(cfg_path.exists())  # nothing persisted on fallback


if __name__ == "__main__":
    unittest.main()
