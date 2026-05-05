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


if __name__ == "__main__":
    unittest.main()
