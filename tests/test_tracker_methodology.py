"""Tests for setup-tracker outcome-measurement correctness fixes:

  #1 episode de-duplication, #3 max-hold time stop, #4 representative-stop
  outcome, #5 stop-first same-bar resolution, #6 cost/slippage in realized R.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402


def _open_scenario(**overrides):
    scenario = {
        "tradeable": True,
        "status": "OPEN",
        "entry_price": 100.0,
        "initial_risk_per_share": 5.0,
        "initial_risk_usd": 500.0,
        "direction": 1.0,
        "shares": 100,
        "remaining_shares": 100,
        "partial_taken": False,
        "partial_shares": 0,
        "realized_pnl": 0.0,
        "realized_r": 0.0,
        "events": [],
        "close_failure_count": 0,
        "close_failure_limit": 2,
        "stop_reference_label": "LOWER_1",
        "active_stop_label": "LOWER_1",
        "partial_target_label": None,
        "final_target_label": "UPPER_3",
        "trail_after_partial_label": None,
        "hard_stop_r_multiple": None,
    }
    scenario.update(overrides)
    return scenario


def _bar(high, low, close):
    return pd.Series({"high": float(high), "low": float(low), "close": float(close)})


class CostModelTests(unittest.TestCase):
    def test_realized_r_is_net_of_round_trip_cost(self):
        scenario = _open_scenario()
        event = m._apply_scenario_exit_event(scenario, 100, 110.0, "2026-01-02", "FINAL_TARGET")
        # gross = (110-100)*100 = 1000; cost = (0.005 + 0.0003*100)*100*2 = 7.0
        self.assertAlmostEqual(event["gross_pnl"], 1000.0)
        self.assertAlmostEqual(event["cost"], 7.0)
        self.assertAlmostEqual(event["pnl"], 993.0)
        self.assertAlmostEqual(scenario["realized_r"], 993.0 / 500.0)

    def test_cost_is_charged_even_on_a_winner(self):
        gross_scenario = _open_scenario()
        m._apply_scenario_exit_event(gross_scenario, 100, 105.0, "2026-01-02", "FINAL_TARGET")
        # net R must be below the gross R of (105-100)*100/500 = 1.0
        self.assertLess(gross_scenario["realized_r"], 1.0)


class StopFirstSameBarTests(unittest.TestCase):
    def test_stop_wins_when_bar_touches_both_stop_and_target(self):
        scenario = _open_scenario(hard_stop_r_multiple=1.25, final_target_label="UPPER_3")
        levels = {"bands": {"UPPER_3": 110.0}}
        # bar high 111 hits target 110; low 93 hits hard stop (100 - 5*1.25 = 93.75)
        events = m._evaluate_tracker_scenario_bar(
            scenario, "LONG", "2026-01-05", _bar(111, 93, 94),
            levels, None, is_entry_day=False, bar_index=2,
        )
        self.assertEqual(scenario["status"], "STOPPED")
        self.assertTrue(any(e["reason"] == "HARD_STOP" for e in events))

    def test_target_still_books_when_stop_not_touched(self):
        scenario = _open_scenario(hard_stop_r_multiple=1.25, final_target_label="UPPER_3")
        levels = {"bands": {"UPPER_3": 110.0}}
        events = m._evaluate_tracker_scenario_bar(
            scenario, "LONG", "2026-01-05", _bar(111, 99, 110),
            levels, None, is_entry_day=False, bar_index=2,
        )
        self.assertEqual(scenario["status"], "TARGET_HIT")
        self.assertTrue(any(e["reason"] == "FINAL_TARGET" for e in events))


class TimeStopTests(unittest.TestCase):
    def test_open_scenario_force_closed_after_max_hold(self):
        scenario = _open_scenario(final_target_label="UPPER_3")
        levels = {"UPPER_3": 200.0}  # target far away, never hit
        # bar before the window: stays open
        m._evaluate_tracker_scenario_bar(
            scenario, "LONG", "2026-01-05", _bar(105, 98, 102),
            levels, None, is_entry_day=False, bar_index=m.TRACKER_MAX_HOLD_DAYS - 1,
        )
        self.assertTrue(m._scenario_is_open(scenario["status"]))
        # bar at the window: time-stopped at the close
        events = m._evaluate_tracker_scenario_bar(
            scenario, "LONG", "2026-01-06", _bar(105, 98, 103),
            levels, None, is_entry_day=False, bar_index=m.TRACKER_MAX_HOLD_DAYS,
        )
        self.assertEqual(scenario["status"], "TIME_STOP")
        self.assertTrue(any(e["reason"] == "TIME_STOP" for e in events))
        self.assertTrue(m._scenario_is_closed(scenario["status"]))


class EpisodeDedupeTests(unittest.TestCase):
    def test_episode_key_ignores_scan_date(self):
        a = {"symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02", "setup_family": "post_earnings_52w_break", "scan_date": "2026-01-05"}
        b = {"symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02", "setup_family": "post_earnings_52w_break", "scan_date": "2026-01-09"}
        self.assertEqual(m._tracker_episode_key(a), m._tracker_episode_key(b))

    def test_dedupe_keeps_one_per_episode_preferring_closed(self):
        rows = [
            {"symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02", "setup_family": "f", "scan_date": "2026-01-05", "closed_setups": 0},
            {"symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02", "setup_family": "f", "scan_date": "2026-01-07", "closed_setups": 1},
            {"symbol": "AMD", "side": "LONG", "anchor_date": "2026-01-03", "setup_family": "f", "scan_date": "2026-01-06", "closed_setups": 0},
        ]
        out = m._dedupe_recent_tracker_family_rows(rows)
        self.assertEqual(len(out), 2)  # one NVDA episode + one AMD episode
        nvda = next(r for r in out if r["symbol"] == "NVDA")
        self.assertEqual(nvda["closed_setups"], 1)  # closed record preferred

    def test_dedupe_prefers_earliest_when_none_closed(self):
        rows = [
            {"symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02", "setup_family": "f", "scan_date": "2026-01-09", "closed_setups": 0},
            {"symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02", "setup_family": "f", "scan_date": "2026-01-05", "closed_setups": 0},
        ]
        out = m._dedupe_recent_tracker_family_rows(rows)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["scan_date"], "2026-01-05")


class RepresentativeStopOutcomeTests(unittest.TestCase):
    def test_representative_uses_primary_stop_not_mean(self):
        # Long primary protective stop is LOWER_1. Give the primary stop a +2R
        # winner and an alternate stop a -1R loser; mean would be +0.5R.
        setup = {
            "side": "LONG",
            "scenarios": {
                "s1": {"tradeable": True, "status": "TARGET_HIT", "total_r": 2.0, "stop_reference_label": "LOWER_1"},
                "s2": {"tradeable": True, "status": "STOPPED", "total_r": -1.0, "stop_reference_label": "SMA_50"},
            },
        }
        outcome = m._summarize_tracker_setup_outcome(setup)
        self.assertAlmostEqual(outcome["avg_closed_r"], 0.5)  # mean across variants
        self.assertAlmostEqual(outcome["representative_closed_r"], 2.0)  # primary stop only
        self.assertEqual(outcome["representative_stop_label"], "LOWER_1")

    def test_representative_falls_back_to_mean_without_primary(self):
        setup = {
            "side": "LONG",
            "scenarios": {
                "s1": {"tradeable": True, "status": "TARGET_HIT", "total_r": 2.0, "stop_reference_label": "SMA_50"},
                "s2": {"tradeable": True, "status": "STOPPED", "total_r": -1.0, "stop_reference_label": "SMA_100"},
            },
        }
        outcome = m._summarize_tracker_setup_outcome(setup)
        self.assertAlmostEqual(outcome["representative_closed_r"], outcome["avg_closed_r"])


if __name__ == "__main__":
    unittest.main()
