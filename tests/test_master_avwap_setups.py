import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap import (  # noqa: E402
    POST_EARNINGS_STOP_FAILURE_CLOSES,
    POST_EARNINGS_STOP_LABEL,
    _build_latest_earnings_release_context,
    _evaluate_tracker_scenario_bar,
    _find_tracker_stop_candidates,
    apply_recent_tracker_setup_family_adjustments,
    build_recent_tracker_setup_family_rows,
    load_scan_earnings_context,
)


def _build_history_with_gap() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=25)
    rows = []
    for idx, dt_value in enumerate(dates):
        base_price = 100.0 + (idx * 0.3)
        open_price = base_price
        close_price = base_price + 0.2
        high_price = close_price + 0.8
        low_price = open_price - 0.8

        if idx == 21:
            open_price = 94.0
            close_price = 93.0
            high_price = 95.0
            low_price = 92.0

        rows.append(
            {
                "datetime": dt_value,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": 1_000_000,
            }
        )

    return pd.DataFrame(rows)


def _build_tracker_setup(
    symbol: str,
    scan_date: str,
    total_r: float,
    *,
    side: str = "LONG",
    priority_bucket: str = "favorite_setup",
    setup_family: str = "avwap_breakout",
    scenario_status: str = "TARGET_HIT",
) -> dict:
    setup_status = "OPEN"
    if scenario_status in {"TARGET_HIT", "STOPPED"}:
        setup_status = "CLOSED"
    return {
        "setup_id": f"{scan_date}:{symbol}:{setup_family}",
        "symbol": symbol,
        "side": side,
        "scan_date": scan_date,
        "priority_bucket": priority_bucket,
        "setup_family": setup_family,
        "setup_status": setup_status,
        "scenarios": {
            "baseline": {
                "experimental": False,
                "tradeable": True,
                "status": scenario_status,
                "total_r": total_r,
            }
        },
    }


class MasterAvwapSetupTests(unittest.TestCase):
    def test_load_scan_earnings_context_reuses_refreshed_earnings_lookup(self):
        earnings_lookup = {"AAPL": ["2026-04-21"]}
        latest_release_map = {
            "AAPL": {
                "earnings_date": "2026-04-21",
                "release_session": "amc",
            }
        }

        with patch("master_avwap.load_or_refresh_earnings", return_value=earnings_lookup) as refresh_mock, patch(
            "master_avwap.load_latest_earnings_release_map",
            return_value=latest_release_map,
        ) as release_mock:
            loaded_earnings, loaded_release_map = load_scan_earnings_context(["aapl", "AAPL"])

        refresh_mock.assert_called_once_with(["AAPL"])
        release_mock.assert_called_once_with(
            ["AAPL"],
            earnings_lookup=earnings_lookup,
        )
        self.assertIs(loaded_earnings, earnings_lookup)
        self.assertIs(loaded_release_map, latest_release_map)

    def test_release_context_uses_datetime_dates_without_trade_date_column(self):
        df = _build_history_with_gap()
        earnings_date = df.iloc[20]["datetime"].date().isoformat()

        context = _build_latest_earnings_release_context(
            df,
            {
                "earnings_date": earnings_date,
                "release_session": "amc",
            },
        )

        self.assertTrue(context["active"])
        self.assertEqual(context["anchor_date"], earnings_date)
        self.assertEqual(context["gap_date"], df.iloc[21]["datetime"].date().isoformat())
        self.assertTrue(context["gap_is_down"])
        self.assertGreater(context["gap_atr_multiple"], 1.0)

    def test_post_earnings_stop_candidate_uses_custom_close_failure_limit(self):
        candidates = _find_tracker_stop_candidates(
            {
                "side": "SHORT",
                "priority_bucket": "favorite_setup",
                "setup_family": "post_earnings_52w_break",
            },
            {
                "current_anchor": {
                    "vwap": 48.0,
                    "bands": {"UPPER_1": 50.0},
                },
                "post_earnings_anchor": {
                    "vwap": 51.5,
                    "bands": {"UPPER_1": 53.0},
                },
                "atr20": 2.0,
            },
        )

        post_stop = next(item for item in candidates if item["label"] == POST_EARNINGS_STOP_LABEL)
        self.assertEqual(post_stop["source_type"], "post_earnings_anchor")
        self.assertEqual(post_stop["level"], 51.5)
        self.assertEqual(post_stop["close_failure_limit"], POST_EARNINGS_STOP_FAILURE_CLOSES)

    def test_post_earnings_dynamic_stop_fails_on_first_close(self):
        scenario = {
            "tradeable": True,
            "status": "OPEN",
            "entry_price": 50.0,
            "initial_risk_per_share": 2.0,
            "initial_risk_usd": 200.0,
            "direction": -1.0,
            "remaining_shares": 100,
            "partial_taken": False,
            "partial_target_label": "",
            "final_target_label": "",
            "stop_reference_label": POST_EARNINGS_STOP_LABEL,
            "active_stop_label": POST_EARNINGS_STOP_LABEL,
            "stop_reference_level": 51.0,
            "close_failure_limit": 1,
            "close_failure_count": 0,
            "hard_stop_r_multiple": None,
            "realized_pnl": 0.0,
            "realized_r": 0.0,
            "unrealized_pnl": 0.0,
            "unrealized_r": 0.0,
            "total_pnl": 0.0,
            "total_r": 0.0,
            "events": [],
        }
        bar_row = pd.Series(
            {
                "open": 50.5,
                "high": 52.5,
                "low": 49.5,
                "close": 51.2,
            }
        )

        events = _evaluate_tracker_scenario_bar(
            scenario,
            "SHORT",
            "2026-02-05",
            bar_row,
            current_anchor_levels=None,
            indicator_row=None,
            is_entry_day=False,
            dynamic_level_overrides={POST_EARNINGS_STOP_LABEL: 51.0},
        )

        self.assertEqual(scenario["status"], "STOPPED")
        self.assertEqual(scenario["close_failure_count"], 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["reason"], "STOP_FAIL")

    def test_recent_tracker_setup_family_rows_favor_recent_winner(self):
        reference_date = date(2026, 4, 22)
        setups = {
            "winner_1": _build_tracker_setup("AAPL", "2026-04-21", 2.1, setup_family="avwap_retest_followthrough"),
            "winner_2": _build_tracker_setup("NVDA", "2026-04-18", 1.8, setup_family="avwap_retest_followthrough"),
            "winner_3": _build_tracker_setup("META", "2026-04-15", 1.6, setup_family="avwap_retest_followthrough"),
            "winner_4": _build_tracker_setup("AMZN", "2026-04-10", 2.4, setup_family="avwap_retest_followthrough"),
            "baseline_1": _build_tracker_setup("MSFT", "2026-04-20", 0.2, setup_family="avwap_breakout"),
            "baseline_2": _build_tracker_setup("CRM", "2026-04-17", -0.1, setup_family="avwap_breakout"),
            "baseline_3": _build_tracker_setup("ORCL", "2026-04-14", 0.0, setup_family="avwap_breakout"),
            "baseline_4": _build_tracker_setup("ADBE", "2026-04-09", 0.3, setup_family="avwap_breakout"),
        }

        recent_rows = build_recent_tracker_setup_family_rows(setups, reference_date=reference_date)
        winning_row = next(
            row for row in recent_rows
            if row["setup_family"] == "avwap_retest_followthrough"
        )
        breakout_row = next(
            row for row in recent_rows
            if row["setup_family"] == "avwap_breakout"
        )

        self.assertGreater(winning_row["score_delta"], 0)
        self.assertGreater(winning_row["avg_closed_r"], winning_row["baseline_avg_closed_r"])
        self.assertGreaterEqual(winning_row["score_delta"], breakout_row["score_delta"])

    def test_recent_tracker_adjustment_boosts_matching_row_once(self):
        reference_date = date(2026, 4, 22)
        tracker_payload = {
            "setups": {
                "winner_1": _build_tracker_setup("AAPL", "2026-04-21", 2.1, setup_family="avwap_retest_followthrough"),
                "winner_2": _build_tracker_setup("NVDA", "2026-04-18", 1.8, setup_family="avwap_retest_followthrough"),
                "winner_3": _build_tracker_setup("META", "2026-04-15", 1.6, setup_family="avwap_retest_followthrough"),
                "winner_4": _build_tracker_setup("AMZN", "2026-04-10", 2.4, setup_family="avwap_retest_followthrough"),
                "baseline_1": _build_tracker_setup("MSFT", "2026-04-20", 0.2, setup_family="avwap_breakout"),
                "baseline_2": _build_tracker_setup("CRM", "2026-04-17", -0.1, setup_family="avwap_breakout"),
                "baseline_3": _build_tracker_setup("ORCL", "2026-04-14", 0.0, setup_family="avwap_breakout"),
                "baseline_4": _build_tracker_setup("ADBE", "2026-04-09", 0.3, setup_family="avwap_breakout"),
            }
        }
        priority_rows = [
            {
                "symbol": "TSLA",
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "setup_family": "avwap_retest_followthrough",
                "score": 100.0,
            },
            {
                "symbol": "GOOG",
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "setup_family": "favorite_zone_watch",
                "score": 100.0,
            },
        ]
        ai_state = {"symbols": {"TSLA": {}, "GOOG": {}}}
        feature_rows_by_symbol = {
            "TSLA": {"priority_score": 100.0},
            "GOOG": {"priority_score": 100.0},
        }

        apply_recent_tracker_setup_family_adjustments(
            priority_rows,
            ai_state,
            feature_rows_by_symbol,
            tracker_payload=tracker_payload,
            reference_date=reference_date,
        )
        boosted_score = priority_rows[0]["score"]

        self.assertGreater(boosted_score, 100.0)
        self.assertTrue(priority_rows[0]["recent_tracker_score_note"])
        self.assertEqual(priority_rows[1]["score"], 100.0)
        self.assertEqual(priority_rows[1]["recent_tracker_score_delta"], 0)
        self.assertEqual(feature_rows_by_symbol["TSLA"]["priority_score"], boosted_score)
        self.assertEqual(ai_state["symbols"]["TSLA"]["priority_score"], boosted_score)

        apply_recent_tracker_setup_family_adjustments(
            priority_rows,
            ai_state,
            feature_rows_by_symbol,
            tracker_payload=tracker_payload,
            reference_date=reference_date,
        )

        self.assertEqual(priority_rows[0]["score"], boosted_score)


if __name__ == "__main__":
    unittest.main()
