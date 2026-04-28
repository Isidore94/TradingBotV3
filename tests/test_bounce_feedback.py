import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
