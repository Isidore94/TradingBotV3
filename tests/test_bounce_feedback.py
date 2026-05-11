import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bounce_bot  # noqa: E402


class BounceFeedbackTests(unittest.TestCase):
    def _build_impulse_retest_bot(self):
        bot = bounce_bot.BounceBot.__new__(bounce_bot.BounceBot)
        bot.atr_cache = {"AAPL": 10.0}
        bot.symbol_metrics = {"AAPL": {"std_vwap": 100.0, "eod_vwap": 120.0}}
        bot.latest_bars = {}
        bot.get_cached_5m_bars = Mock(return_value=[])
        bot.get_symbol_direction = Mock(return_value="long")
        bot.is_bounce_type_enabled = Mock(return_value=True)
        bot._build_intraday_level_series_maps = Mock(return_value={})
        bot._impulse_regime_transition_ok = Mock(return_value=True)
        return bot

    def _build_impulse_retest_frame(self, *, current_low: float) -> pd.DataFrame:
        timestamps = pd.date_range("2026-05-06 09:30:00", periods=13, freq="5min")
        rows = [
            (101.0, 102.0, 100.0, 101.0),
            (101.0, 106.0, 100.5, 105.0),
            (105.0, 112.0, 104.0, 111.0),
            (111.0, 119.0, 110.0, 118.0),
            (118.0, 125.0, 117.0, 124.0),
            (123.0, 124.0, 119.0, 120.0),
            (120.0, 122.0, 116.0, 117.0),
            (117.0, 119.0, 113.0, 114.0),
            (114.0, 117.0, 112.0, 113.0),
            (113.0, 116.0, 111.5, 114.0),
            (114.0, 117.0, 112.0, 116.0),
            (116.0, 119.0, 113.0, 118.0),
            (119.5, 122.0, current_low, 121.0),
        ]
        return pd.DataFrame(
            [
                {
                    "datetime": ts,
                    "time": ts.strftime("%Y%m%d  %H:%M:%S"),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 10_000 + idx,
                }
                for idx, (ts, (open_, high, low, close)) in enumerate(zip(timestamps, rows))
            ]
        )

    def test_record_bounce_feedback_writes_human_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "intraday_bounce_feedback.csv"
            context = {
                "event_id": "AAPL_long_20260424_vwap",
                "trade_date": "2026-04-24",
                "symbol": "AAPL",
                "direction": "long",
                "bounce_types": "impulse_retest_vwap_eod;vwap",
                "alert_message": "AAPL: Bounce confirmed (long) from ['impulse_retest_vwap_eod', 'vwap']",
            }

            with patch.object(bounce_bot, "INTRADAY_BOUNCE_FEEDBACK_CSV", feedback_path):
                written_path = bounce_bot.record_bounce_feedback(
                    context,
                    "issue",
                    "Impulse retest never actually got back to VWAP/EOD AVWAP.",
                    source="test",
                )

            self.assertEqual(written_path, feedback_path)
            rows = pd.read_csv(feedback_path)
            self.assertEqual(rows.loc[0, "symbol"], "AAPL")
            self.assertEqual(rows.loc[0, "rating"], "issue")
            self.assertIn("VWAP/EOD", rows.loc[0, "reason"])

    def test_impulse_retest_vwap_eod_rejects_middle_of_wide_vwap_gap(self):
        bot = self._build_impulse_retest_bot()
        df = self._build_impulse_retest_frame(current_low=110.0)

        candidate = bot.evaluate_bounce_candidate(
            "AAPL",
            df,
            allowed_bounce_types={"impulse_retest_vwap_eod"},
        )

        self.assertIsNone(candidate)

    def test_impulse_retest_vwap_eod_accepts_actual_eod_vwap_retest(self):
        bot = self._build_impulse_retest_bot()
        df = self._build_impulse_retest_frame(current_low=119.0)

        candidate = bot.evaluate_bounce_candidate(
            "AAPL",
            df,
            allowed_bounce_types={"impulse_retest_vwap_eod"},
        )

        self.assertIsNotNone(candidate)
        self.assertIn("impulse_retest_vwap_eod", candidate["triggered_levels"])
        self.assertIn("eod_vwap", candidate["levels"])
        self.assertNotIn("vwap", candidate["levels"])
        self.assertTrue(candidate["confirm_immediately"])

    def test_d1_flags_prime_without_startup_gui_spam(self):
        bot = bounce_bot.BounceBot.__new__(bounce_bot.BounceBot)
        bot.emitted_master_avwap_d1_flags = set()
        bot.master_avwap_d1_flags_primed_date = None
        bot.gui_callback = Mock()
        bot.log_symbol = Mock()

        first_event = {
            "symbol": "TSN",
            "side": "LONG",
            "direction": "long",
            "event_type": "breakout_5d",
            "label": "5D breakout",
            "reason": "D1 breakout over recent range",
            "priority_score": 26,
            "source": "focus",
        }
        second_event = {
            **first_event,
            "symbol": "WFC",
            "side": "SHORT",
            "direction": "short",
            "priority_score": 25,
        }
        bot._build_master_avwap_d1_flag_events = Mock(return_value=[first_event])

        bot.emit_master_avwap_d1_flags()

        bot.gui_callback.assert_not_called()
        self.assertEqual(len(bot.emitted_master_avwap_d1_flags), 1)

        bot._build_master_avwap_d1_flag_events.return_value = [first_event, second_event]
        bot.emit_master_avwap_d1_flags()

        bot.gui_callback.assert_called_once()
        message, tag = bot.gui_callback.call_args.args
        self.assertIn("MASTER_AVWAP_D1_FLAG: WFC", message)
        self.assertEqual(tag, "d1_flag_short")
        self.assertEqual(len(bot.emitted_master_avwap_d1_flags), 2)

    def test_d1_preloaded_trigger_emits_on_intraday_cross_once(self):
        bot = bounce_bot.BounceBot.__new__(bounce_bot.BounceBot)
        bot.master_avwap_d1_watchlist = {
            "AAPL": {
                "symbol": "AAPL",
                "side": "LONG",
                "direction": "long",
                "active_current_scan": True,
                "priority_score": 240,
                "watchlist_run_date": "2026-05-05",
                "trigger_levels": [
                    {
                        "trigger_id": "first_dev_break:UPPER_1:102.0000",
                        "label": "UPPER_1",
                        "level": 102.0,
                        "action": "break_above",
                        "event_type": "first_dev_break",
                        "alert_label": "1st-dev break",
                        "reason": "Armed from AVWAPE-to-UPPER_1 zone.",
                        "source": "favorite_zone",
                        "armed_price": 101.0,
                    }
                ],
            }
        }
        bot.emitted_master_avwap_d1_flags = set()
        bot.gui_callback = Mock()
        bot.log_symbol = Mock()

        today_df = pd.DataFrame(
            [
                {
                    "datetime": pd.Timestamp("2026-05-06 09:35:00"),
                    "open": 101.1,
                    "high": 101.8,
                    "low": 100.9,
                    "close": 101.6,
                    "volume": 1000,
                    "time": "20260506  09:35:00",
                },
                {
                    "datetime": pd.Timestamp("2026-05-06 09:40:00"),
                    "open": 101.7,
                    "high": 102.4,
                    "low": 101.5,
                    "close": 102.3,
                    "volume": 1200,
                    "time": "20260506  09:40:00",
                },
            ]
        )

        self.assertEqual(bot.emit_master_avwap_intraday_trigger_flags("AAPL", today_df), 1)
        message, tag = bot.gui_callback.call_args.args
        self.assertIn("MASTER_AVWAP_D1_FLAG: AAPL", message)
        self.assertIn("UPPER_1@102.00", message)
        self.assertEqual(tag, "d1_flag_long")

        self.assertEqual(bot.emit_master_avwap_intraday_trigger_flags("AAPL", today_df), 0)
        bot.gui_callback.assert_called_once()

    def test_d1_preloaded_trigger_does_not_replay_earlier_intraday_cross(self):
        bot = bounce_bot.BounceBot.__new__(bounce_bot.BounceBot)
        bot.master_avwap_d1_watchlist = {
            "AAPL": {
                "symbol": "AAPL",
                "side": "LONG",
                "direction": "long",
                "active_current_scan": True,
                "priority_score": 240,
                "watchlist_run_date": "2026-05-05",
                "trigger_levels": [
                    {
                        "trigger_id": "first_dev_break:UPPER_1:102.0000",
                        "label": "UPPER_1",
                        "level": 102.0,
                        "action": "break_above",
                        "event_type": "first_dev_break",
                        "alert_label": "1st-dev break",
                        "armed_price": 101.0,
                    }
                ],
            }
        }
        bot.emitted_master_avwap_d1_flags = set()
        bot.gui_callback = Mock()
        bot.log_symbol = Mock()

        today_df = pd.DataFrame(
            [
                {
                    "datetime": pd.Timestamp("2026-05-06 09:35:00"),
                    "open": 101.1,
                    "high": 102.4,
                    "low": 100.9,
                    "close": 102.2,
                    "volume": 1000,
                    "time": "20260506  09:35:00",
                },
                {
                    "datetime": pd.Timestamp("2026-05-06 09:40:00"),
                    "open": 102.2,
                    "high": 102.6,
                    "low": 102.0,
                    "close": 102.4,
                    "volume": 1200,
                    "time": "20260506  09:40:00",
                },
            ]
        )

        self.assertEqual(bot.emit_master_avwap_intraday_trigger_flags("AAPL", today_df), 0)
        bot.gui_callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
