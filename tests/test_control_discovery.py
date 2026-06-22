"""Tests for the control / holdout discovery system (#2):

selection of rejected setups, episode-deduped discovery comparison, and pruning.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402


def _priority_row(symbol, score, *, bucket="tracked", signal=True, side="LONG", blocked=False):
    return {
        "symbol": symbol,
        "side": side,
        "priority_bucket": bucket,
        "score": score,
        "has_favorite_signal": signal,
        "ranking_blocked": blocked,
    }


def _closed_control_setup(symbol, family, closed_r, reason, *, anchor="2026-01-02", scan="2026-01-03", side="LONG"):
    return {
        "symbol": symbol,
        "side": side,
        "anchor_date": anchor,
        "scan_date": scan,
        "setup_family": family,
        "is_control": True,
        "control_reason": reason,
        "scenarios": {
            "s1": {
                "tradeable": True,
                "status": "TARGET_HIT" if closed_r > 0 else "STOPPED",
                "total_r": closed_r,
                "stop_reference_label": "LOWER_1",  # primary stop for LONG
            }
        },
    }


class SelectControlRowsTests(unittest.TestCase):
    def test_excludes_favorites_tracked_and_non_setups(self):
        fav = _priority_row("FAV", 150, bucket="favorite_setup")
        near = _priority_row("NEAR", 120, bucket="near_favorite_zone")
        nosig = _priority_row("NOSIG", 95, signal=False)
        blocked = _priority_row("BLK", 95, blocked=True)
        real = _priority_row("REAL", 90)
        rows = [fav, near, nosig, blocked, real]
        selected = m.select_tracker_control_rows(rows, [fav, near], scan_date="2026-06-21")
        symbols = {r["symbol"] for r in selected}
        self.assertEqual(symbols, {"REAL"})
        self.assertEqual(selected[0]["control_reason"], "near_miss")  # 90 >= 100-25

    def test_near_miss_vs_random_stratification_and_cap(self):
        rows = [_priority_row(f"S{i}", 95 - i * 5) for i in range(12)]  # 95..40
        selected = m.select_tracker_control_rows(rows, [], scan_date="2026-06-21", max_rows=6)
        self.assertEqual(len(selected), 6)
        reasons = [r["control_reason"] for r in selected]
        # near-miss = score >= 75; those come first
        self.assertEqual(reasons[:4], ["near_miss"] * 4)
        self.assertIn("random", reasons)

    def test_deterministic_for_same_scan_date(self):
        rows = [_priority_row(f"S{i}", 60) for i in range(10)]
        a = m.select_tracker_control_rows(rows, [], scan_date="2026-06-21", max_rows=4)
        b = m.select_tracker_control_rows(rows, [], scan_date="2026-06-21", max_rows=4)
        self.assertEqual([r["symbol"] for r in a], [r["symbol"] for r in b])

    def test_disabled_flag_returns_empty(self):
        rows = [_priority_row("REAL", 90)]
        with patch.object(m, "TRACKER_CONTROL_SAMPLING_ENABLED", False):
            self.assertEqual(m.select_tracker_control_rows(rows, [], scan_date="x"), [])


class ControlDiscoveryTests(unittest.TestCase):
    def test_outperforming_family_is_flagged_and_ranked_first(self):
        control = {}
        for i in range(6):
            s = _closed_control_setup(f"W{i}", "post_earnings_52w_break", 1.2, "random", anchor=f"2026-01-{i+1:02d}")
            control[f"control:w{i}"] = s
        for i in range(6):
            s = _closed_control_setup(f"L{i}", "sma_breakout_50_reclaim", -0.8, "random", anchor=f"2026-02-{i+1:02d}")
            control[f"control:l{i}"] = s
        tracker = {"setups": {}, "control_setups": control}

        discovery = m.build_control_discovery_rows(tracker)
        families = discovery["families"]
        self.assertGreaterEqual(len(families), 2)
        top = families[0]
        self.assertAlmostEqual(top["avg_closed_r"], 1.2, places=2)
        self.assertTrue(top["outperforming"])
        worst = families[-1]
        self.assertLess(worst["avg_closed_r"], 0)
        self.assertFalse(worst["outperforming"])

    def test_episode_dedupe_counts_one_per_episode(self):
        # Same episode (symbol/side/anchor/family) re-scanned twice => 1 episode.
        control = {
            "control:a": _closed_control_setup("NVDA", "post_earnings_52w_break", 1.0, "random", anchor="2026-01-02", scan="2026-01-03"),
            "control:b": _closed_control_setup("NVDA", "post_earnings_52w_break", 1.0, "random", anchor="2026-01-02", scan="2026-01-06"),
        }
        discovery = m.build_control_discovery_rows({"setups": {}, "control_setups": control})
        total_episodes = sum(row["closed_episodes"] for row in discovery["families"])
        self.assertEqual(total_episodes, 1)

    def test_cohort_comparison_separates_promoted_and_control(self):
        promoted = {
            "p1": _closed_control_setup("AAA", "post_earnings_52w_break", 1.5, "promoted", anchor="2026-01-02"),
        }
        # strip control markers from promoted (they live in the favorite namespace)
        promoted["p1"].pop("is_control", None)
        promoted["p1"].pop("control_reason", None)
        control = {
            "control:c1": _closed_control_setup("BBB", "post_earnings_52w_break", -0.5, "near_miss", anchor="2026-01-03"),
        }
        discovery = m.build_control_discovery_rows({"setups": promoted, "control_setups": control})
        cohorts = {row["cohort"]: row for row in discovery["cohorts"]}
        self.assertEqual(cohorts["promoted"]["closed_episodes"], 1)
        self.assertAlmostEqual(cohorts["promoted"]["avg_closed_r"], 1.5, places=2)
        self.assertEqual(cohorts["near_miss"]["closed_episodes"], 1)
        self.assertAlmostEqual(cohorts["near_miss"]["avg_closed_r"], -0.5, places=2)

    def test_report_writes_file(self):
        control = {
            f"control:w{i}": _closed_control_setup(f"W{i}", "post_earnings_52w_break", 1.2, "random", anchor=f"2026-01-{i+1:02d}")
            for i in range(6)
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "discovery.txt"
            m.write_control_discovery_report(path, {"setups": {}, "control_setups": control})
            text = path.read_text(encoding="utf-8")
        self.assertIn("control / holdout discovery", text)
        self.assertIn("DISCOVER", text)  # the outperforming family is flagged


class PruneControlSetupsTests(unittest.TestCase):
    def test_drops_records_older_than_keep_days(self):
        control = {
            "old": {"symbol": "A", "scan_date": "2025-01-01"},
            "fresh": {"symbol": "B", "scan_date": "2026-06-10"},
        }
        m._prune_control_setups(control, reference_scan_date="2026-06-21")
        self.assertNotIn("old", control)
        self.assertIn("fresh", control)

    def test_caps_total_record_count(self):
        control = {
            f"c{i}": {"symbol": f"S{i}", "scan_date": f"2026-06-{(i % 28) + 1:02d}"}
            for i in range(m.CONTROL_SETUP_MAX_RECORDS + 25)
        }
        m._prune_control_setups(control, reference_scan_date="2026-06-21")
        self.assertLessEqual(len(control), m.CONTROL_SETUP_MAX_RECORDS)


if __name__ == "__main__":
    unittest.main()
