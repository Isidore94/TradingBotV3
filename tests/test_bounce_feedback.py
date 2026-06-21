import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

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

    def test_master_avwap_events_today_reuses_file_cache_until_signals_change(self):
        bot = bounce_bot.BounceBot.__new__(bounce_bot.BounceBot)
        bot.master_avwap_events = {}
        bot.master_avwap_last_scan_date = None
        bot._master_avwap_events_cache_key = None

        with tempfile.TemporaryDirectory() as temp_dir:
            signals_path = Path(temp_dir) / "avwap_signals.csv"
            signals_path.write_text("header\nfirst\n", encoding="utf-8")

            with patch.object(bounce_bot, "MASTER_AVWAP_SIGNALS_FILENAME", signals_path):
                with patch.object(
                    bounce_bot,
                    "load_master_avwap_events_for_date",
                    side_effect=[{"AAPL": []}, {"TSLA": []}],
                ) as load_mock:
                    bot.load_master_avwap_events_today()
                    bot.load_master_avwap_events_today()

                    self.assertEqual(load_mock.call_count, 1)
                    self.assertEqual(bot.master_avwap_events, {"AAPL": []})

                    signals_path.write_text("header\nchanged\nwith_more_bytes\n", encoding="utf-8")
                    bot.load_master_avwap_events_today()

                    self.assertEqual(load_mock.call_count, 2)
                    self.assertEqual(bot.master_avwap_events, {"TSLA": []})

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
