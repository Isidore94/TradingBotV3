import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bounce_bot  # noqa: E402


def _flat_h1_bars(*, direction="long"):
    start = datetime(2026, 5, 29, 9, 30)
    bars = []
    for idx in range(21):
        open_price = 100.0
        high_price = 100.50
        low_price = 99.50
        close_price = 100.0
        if idx == 19 and direction == "long":
            high_price = 100.35
            low_price = 99.98
        elif idx == 19:
            high_price = 100.02
            low_price = 99.65
        elif idx == 20 and direction == "long":
            high_price = 100.90
            low_price = 100.20
            close_price = 100.80
        elif idx == 20:
            high_price = 99.80
            low_price = 99.10
            close_price = 99.20

        bars.append(
            bounce_bot.IbBar(
                dt=start + timedelta(hours=idx),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
            )
        )
    return bars


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

    def test_detect_mid_earnings_h1_long_bounce(self):
        bars = _flat_h1_bars(direction="long")
        result = bounce_bot.detect_mid_earnings_h1_bounce(
            bars,
            "long",
            atr20=2.0,
            reference_date=bars[-1].dt.date(),
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["triggered_levels"], ["h1_ema_15", "h1_sma_20"])
        self.assertTrue(result["confirm_immediately"])
        self.assertIn("closed above", result["reason"])

    def test_detect_mid_earnings_h1_short_bounce(self):
        bars = _flat_h1_bars(direction="short")
        result = bounce_bot.detect_mid_earnings_h1_bounce(
            bars,
            "short",
            atr20=2.0,
            reference_date=bars[-1].dt.date(),
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["triggered_levels"], ["h1_ema_15", "h1_sma_20"])
        self.assertIn("closed below", result["reason"])

    def test_detect_mid_earnings_h1_requires_confirmation_close(self):
        bars = _flat_h1_bars(direction="long")
        bars[-1] = bounce_bot.IbBar(
            dt=bars[-1].dt,
            open=100.0,
            high=100.30,
            low=99.70,
            close=99.95,
        )

        result = bounce_bot.detect_mid_earnings_h1_bounce(
            bars,
            "long",
            atr20=2.0,
            reference_date=bars[-1].dt.date(),
        )

        self.assertIsNone(result)

    def test_intraday_bounce_performance_uses_confirmed_outcomes_and_clips_r(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            candidates_path = temp_path / "intraday_bounce_candidates.csv"
            outcomes_path = temp_path / "intraday_bounce_outcomes.csv"
            report_path = temp_path / "intraday_bounce_performance.txt"
            pd.DataFrame(
                [
                    {
                        "event_id": "AAPL_long_1",
                        "event_type": "confirmed",
                        "logged_at": "2026-06-01T09:35:00",
                        "trade_date": "2026-06-01",
                        "symbol": "AAPL",
                        "direction": "long",
                        "bounce_types": "vwap;ema_8",
                        "score": 44,
                        "risk_per_share": 0.40,
                        "rrs_spy": 2.5,
                        "market_environment": "bullish",
                    },
                    {
                        "event_id": "MSFT_long_1",
                        "event_type": "confirmed",
                        "logged_at": "2026-06-01T10:10:00",
                        "trade_date": "2026-06-01",
                        "symbol": "MSFT",
                        "direction": "long",
                        "bounce_types": "vwap",
                        "score": 40,
                        "risk_per_share": 0.35,
                        "rrs_spy": 1.1,
                        "market_environment": "bullish",
                    },
                    {
                        "event_id": "NVDA_long_near",
                        "event_type": "near_miss",
                        "logged_at": "2026-06-01T10:15:00",
                        "trade_date": "2026-06-01",
                        "symbol": "NVDA",
                        "direction": "long",
                        "bounce_types": "vwap",
                        "score": 35,
                        "risk_per_share": 0.50,
                        "rrs_spy": 3.0,
                        "market_environment": "bullish",
                    },
                ]
            ).to_csv(candidates_path, index=False)
            pd.DataFrame(
                [
                    {
                        "event_id": "AAPL_long_1",
                        "event_type": "final",
                        "logged_at": "2026-06-01T10:35:00",
                        "entry_time": "2026-06-01T09:35:00",
                        "bars_elapsed": 12,
                        "close_r": 20.0,
                        "mfe_r": 22.0,
                        "mae_r": -0.20,
                        "target_1r_hit": True,
                        "target_2r_hit": True,
                        "stop_hit": False,
                        "status": "target2_seen",
                    },
                    {
                        "event_id": "MSFT_long_1",
                        "event_type": "final",
                        "logged_at": "2026-06-01T11:10:00",
                        "entry_time": "2026-06-01T10:10:00",
                        "bars_elapsed": 12,
                        "close_r": 1.0,
                        "mfe_r": 1.5,
                        "mae_r": -0.10,
                        "target_1r_hit": True,
                        "target_2r_hit": False,
                        "stop_hit": False,
                        "status": "complete",
                    },
                    {
                        "event_id": "NVDA_long_near",
                        "event_type": "final",
                        "logged_at": "2026-06-01T11:15:00",
                        "entry_time": "2026-06-01T10:15:00",
                        "bars_elapsed": 12,
                        "close_r": -2.0,
                        "mfe_r": 0.1,
                        "mae_r": -2.0,
                        "target_1r_hit": False,
                        "target_2r_hit": False,
                        "stop_hit": True,
                        "status": "stop_seen",
                    },
                ]
            ).to_csv(outcomes_path, index=False)

            rows = bounce_bot.build_intraday_bounce_performance_rows(
                candidates_path=candidates_path,
                outcomes_path=outcomes_path,
                min_samples=2,
            )
            bounce_bot.write_intraday_bounce_performance_report(rows, report_path=report_path)
            report_text = report_path.read_text(encoding="utf-8")

        vwap_row = next(
            row for row in rows
            if row["dimension"] == "bounce_type"
            and row["direction"] == "long"
            and row["segment"] == "vwap"
        )
        self.assertEqual(vwap_row["sample_count"], 2)
        self.assertAlmostEqual(vwap_row["avg_close_r"], 2.5)
        self.assertEqual(vwap_row["recommendation"], "boost")
        self.assertIn("Potential score boosts", report_text)


if __name__ == "__main__":
    unittest.main()
