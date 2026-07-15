import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bounce_bot  # noqa: E402


class _FakeSessionWindow:
    def __init__(self, reference):
        ref = reference if isinstance(reference, datetime) else datetime(2026, 6, 1, 10, 0)
        tz = timezone(timedelta(hours=-7))
        self.open_local = datetime(ref.year, ref.month, ref.day, 6, 30, tzinfo=tz)
        self.close_local = datetime(ref.year, ref.month, ref.day, 13, 0, tzinfo=tz)


def _fake_session_window(reference=None, local_timezone_name=None):
    return _FakeSessionWindow(reference)


def _test_bounce_bot_with_state(state):
    bot = object.__new__(bounce_bot.BounceBot)
    bot.pending_bounce_outcomes = {state["event_id"]: dict(state)}
    bot._save_pending_bounce_outcomes = lambda: None
    return bot


def _pending_state(event_id="AAPL_long_1", *, symbol="AAPL", direction="long", risk=1.0):
    entry_price = 100.0
    if direction == "long":
        stop_price = entry_price - risk
        target_1r = entry_price + risk
        target_2r = entry_price + (2 * risk)
    else:
        stop_price = entry_price + risk
        target_1r = entry_price - risk
        target_2r = entry_price - (2 * risk)
    return {
        "event_id": event_id,
        "symbol": symbol,
        "direction": direction,
        "trade_date": "2026-06-01",
        "entry_time": "2026-06-01T10:00:00",
        "entry_price": entry_price,
        "stop_price": stop_price,
        "risk_per_share": risk,
        "target_1r": target_1r,
        "target_2r": target_2r,
        "milestones_logged": [],
        "outcome_mode": "eod_hold",
        "context": {},
    }


def _intraday_rows(*rows):
    return pd.DataFrame(
        [
            {
                "time": dt.strftime("%Y%m%d  %H:%M:%S"),
                "datetime": dt,
                "high": high,
                "low": low,
                "close": close,
            }
            for dt, high, low, close in rows
        ]
    )


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
    def test_bounce_candidate_industry_rrs_bonus_is_directional(self):
        bot = bounce_bot.BounceBot.__new__(bounce_bot.BounceBot)
        levels = {"vwap": {}}

        base = bot._score_bounce_candidate_snapshot(
            "long",
            levels,
            {"rrs_spy": "", "rrs_sector": "", "rrs_industry": ""},
        )
        aligned_long = bot._score_bounce_candidate_snapshot(
            "long",
            levels,
            {"rrs_spy": "", "rrs_sector": "", "rrs_industry": 3.0},
        )
        counter_long = bot._score_bounce_candidate_snapshot(
            "long",
            levels,
            {"rrs_spy": "", "rrs_sector": "", "rrs_industry": -3.0},
        )
        aligned_short = bot._score_bounce_candidate_snapshot(
            "short",
            levels,
            {"rrs_spy": "", "rrs_sector": "", "rrs_industry": -3.0},
        )

        self.assertEqual(aligned_long, base + 6.0)
        self.assertEqual(counter_long, base)
        self.assertEqual(aligned_short, base + 6.0)

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

    def test_d1_upgrade_watch_formats_as_research_only(self):
        bot = bounce_bot.BounceBot.__new__(bounce_bot.BounceBot)

        message = bot._format_master_avwap_d1_flag_event(
            {
                "symbol": "MSFT",
                "direction": "long",
                "label": "1st-dev break",
                "source": "watchlist_upgrade_target",
                "target_tier": "A/S",
                "trigger_id": "first_dev_break:UPPER_1:103.0000",
                "level_label": "UPPER_1",
                "level": 103.0,
                "reason": "A/S upgrade target: clear UPPER_1 resistance.",
            }
        )

        self.assertIn("MASTER_AVWAP_D1_RESEARCH: MSFT", message)
        self.assertIn("Developing level observation: 1st-dev break", message)
        self.assertIn("status=research-only", message)
        self.assertNotIn("A/S upgrade", message)

    def test_d1_bucket_upgrade_uses_bucket_prefix(self):
        bot = bounce_bot.BounceBot.__new__(bounce_bot.BounceBot)

        message = bot._format_master_avwap_d1_flag_event(
            {
                "symbol": "NVDA",
                "direction": "long",
                "label": "Favorite setup upgrade",
                "reason": "Near favorite zone -> Favorite setup",
                "source": "bucket_upgrade",
                "priority_score": 245,
            }
        )

        self.assertIn("MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA", message)
        self.assertIn("Favorite setup upgrade", message)

    def test_d1_flag_builder_only_uses_bucket_upgrade_alerts(self):
        bot = bounce_bot.BounceBot.__new__(bounce_bot.BounceBot)
        bot.master_avwap_focus_map = {
            "AAPL": {
                "symbol": "AAPL",
                "side": "LONG",
                "priority_score": 220,
                "mid_earnings_ema15_trigger": True,
            }
        }
        bot.master_avwap_events = {
            "MSFT": [
                {
                    "symbol": "MSFT",
                    "signal_type": "CROSS_UP_VWAP",
                    "side": "LONG",
                    "level": "VWAP",
                }
            ]
        }
        bot.master_avwap_d1_watchlist = {
            "TSLA": {
                "symbol": "TSLA",
                "side": "SHORT",
                "active_current_scan": True,
                "upgrade_targets": [
                    {
                        "label": "LOWER_1",
                        "level": 90.0,
                        "action": "break_below",
                    }
                ],
            }
        }
        bot.master_avwap_d1_upgrade_alerts = {
            "OLD": {
                "symbol": "OLD",
                "side": "LONG",
                "upgrade_targets": [
                    {
                        "label": "UPPER_1",
                        "level": 103.0,
                        "action": "break_above",
                    }
                ],
            },
            "NVDA": {
                "symbol": "NVDA",
                "side": "LONG",
                "bucket_upgrade_events": [
                    {
                        "symbol": "NVDA",
                        "side": "LONG",
                        "event_type": "bucket_upgrade",
                        "label": "High conviction upgrade",
                        "reason": "Favorite setup -> High conviction",
                        "source": "bucket_upgrade",
                        "previous_bucket": "favorite_setup",
                        "priority_bucket": "high_conviction",
                    }
                ],
            },
        }

        events = bot._build_master_avwap_d1_flag_events()

        self.assertEqual([event["symbol"] for event in events], ["NVDA"])
        self.assertEqual(events[0]["source"], "bucket_upgrade")

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

    def test_d1_upgrade_wick_is_logged_as_research_without_gui_alert(self):
        bot = bounce_bot.BounceBot.__new__(bounce_bot.BounceBot)
        watch_event = {
            "symbol": "MSFT",
            "side": "LONG",
            "direction": "long",
            "event_type": "first_dev_break",
            "label": "A/S upgrade: 1st-dev break",
            "reason": "A/S upgrade target: clear UPPER_1 resistance.",
            "source": "watchlist_upgrade_target",
            "trigger_id": "first_dev_break:UPPER_1:103.0000",
            "level_label": "UPPER_1",
            "level": 103.0,
            "target_tier": "A/S",
        }
        bot._build_master_avwap_d1_flag_events = Mock(return_value=[watch_event])
        bot.master_avwap_d1_flags_primed_date = None
        bot.emitted_master_avwap_d1_flags = set()
        bot.gui_callback = Mock()
        bot.log_symbol = Mock()

        bot.emit_master_avwap_d1_flags()

        bot.gui_callback.assert_not_called()
        bot.master_avwap_d1_upgrade_alerts = {
            "MSFT": {
                "symbol": "MSFT",
                "side": "LONG",
                "direction": "long",
                "active_current_scan": True,
                "priority_score": 185,
                "watchlist_run_date": "2026-05-05",
                "trigger_levels": [
                    {
                        "trigger_id": "first_dev_break:UPPER_1:103.0000",
                        "label": "UPPER_1",
                        "level": 103.0,
                        "action": "break_above",
                        "event_type": "first_dev_break",
                        "alert_label": "1st-dev break",
                        "reason": "A/S upgrade target: clear UPPER_1 resistance.",
                        "armed_price": 102.0,
                        "target_tier": "A/S",
                        "upgrade_only": True,
                    }
                ],
            }
        }
        bot.master_avwap_d1_watchlist = {}
        today_df = pd.DataFrame(
            [
                {
                    "datetime": pd.Timestamp("2026-05-06 09:35:00"),
                    "open": 102.1,
                    "high": 102.8,
                    "low": 101.9,
                    "close": 102.6,
                    "volume": 1000,
                    "time": "20260506  09:35:00",
                },
                {
                    "datetime": pd.Timestamp("2026-05-06 09:40:00"),
                    "open": 102.7,
                    "high": 103.4,
                    "low": 102.5,
                    # A wick reached the armed level but the bar did not close
                    # through it. This remains useful research evidence; it is
                    # not an A/S upgrade or an actionable GUI alert.
                    "close": 102.9,
                    "volume": 1200,
                    "time": "20260506  09:40:00",
                },
            ]
        )

        self.assertEqual(bot.emit_master_avwap_intraday_trigger_flags("MSFT", today_df), 1)
        bot.gui_callback.assert_not_called()
        bot.log_symbol.assert_called_once()
        logged_symbol, message = bot.log_symbol.call_args.args
        self.assertEqual(logged_symbol, "MSFT")
        self.assertIn("MASTER_AVWAP_D1_RESEARCH: MSFT", message)
        self.assertIn("Developing level observation: 1st-dev break", message)
        self.assertIn("price=102.90", message)
        self.assertIn("status=research-only", message)
        self.assertNotIn("A/S upgrade", message)

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

    def test_top_pattern_h1_entry_uses_long_only_h1_ema15(self):
        bars = _flat_h1_bars(direction="long")
        bot = object.__new__(bounce_bot.BounceBot)
        bot.master_avwap_focus_map = {
            "TOPS": {
                "symbol": "TOPS",
                "side": "LONG",
                "priority_bucket": "top_strength_watchlist",
                "setup_family": "top_pattern_tracking",
                "top_pattern_watch": True,
            }
        }

        result = bounce_bot.detect_mid_earnings_h1_bounce(
            bars,
            "long",
            atr20=2.0,
            reference_date=bars[-1].dt.date(),
            allowed_level_keys=bounce_bot.H1_TOP_PATTERN_ENTRY_LEVELS,
        )

        self.assertTrue(bot._is_top_pattern_h1_entry_focus("TOPS", "long"))
        self.assertFalse(bot._is_top_pattern_h1_entry_focus("TOPS", "short"))
        self.assertIsNotNone(result)
        self.assertEqual(result["triggered_levels"], ["h1_ema_15"])

    def test_bounce_type_learning_helpers_normalize_and_merge_all_levels(self):
        self.assertEqual(
            bounce_bot._split_bounce_type_text("10_candle_low;vwap;10_candle_high"),
            ["10_candle", "vwap"],
        )

        bot = object.__new__(bounce_bot.BounceBot)
        merged = bot._merge_bounce_candidate_learning_levels(
            {
                "levels": {"vwap": 100.0},
                "triggered_levels": ["vwap"],
            },
            {
                "levels": {"ema_21": 99.0, "10_candle_low": 98.0},
                "triggered_levels": ["ema_21", "10_candle_low"],
            },
        )

        self.assertEqual(set(merged["levels"]), {"vwap", "ema_21", "10_candle_low"})
        self.assertEqual(merged["triggered_levels"], ["vwap", "ema_21", "10_candle_low"])
        self.assertEqual(merged["learning_bounce_types"], ["10_candle", "ema_21", "vwap"])

    def test_master_avwap_swing_trait_tags_include_swing_context(self):
        bot = object.__new__(bounce_bot.BounceBot)
        tags = bot._master_avwap_swing_trait_tags(
            {
                "preferred_swing_focus": True,
                "top_pattern_entry": True,
                "post_earnings_active": False,
                "priority_bucket": "favorite_setup",
                "setup_family": "top_pattern",
            }
        )

        self.assertIn("preferred_swing_focus", tags)
        self.assertIn("top_pattern_entry", tags)
        self.assertIn("favorite_setup", tags)
        self.assertIn("family:top_pattern", tags)
        self.assertNotIn("post_earnings_active", tags)

    def test_human_focus_bounce_policy_bypasses_disabled_types_for_matching_side(self):
        bot = object.__new__(bounce_bot.BounceBot)
        bot.human_focus_map = {"long": {"AAPL"}, "short": {"TSLA"}}
        enabled_types = {"ema_8", "vwap"}

        allowed_types, include_disabled = bot._bounce_eval_filter_options_for_symbol(
            "AAPL",
            "long",
            enabled_types,
        )
        self.assertIsNone(allowed_types)
        self.assertTrue(include_disabled)

        allowed_types, include_disabled = bot._bounce_eval_filter_options_for_symbol(
            "AAPL",
            "short",
            enabled_types,
        )
        self.assertEqual(allowed_types, enabled_types)
        self.assertFalse(include_disabled)

    def test_human_focus_symbols_are_scanned_and_set_direction(self):
        bot = object.__new__(bounce_bot.BounceBot)
        bot.longs = []
        bot.shorts = []
        bot.master_avwap_d1_watchlist = {}
        bot.master_avwap_d1_upgrade_alerts = {}
        bot.master_avwap_focus_map = {}
        bot.latest_scan_extremes = {}
        bot.human_focus_map = {"long": {"AAPL"}, "short": {"TSLA"}}

        self.assertEqual(bot.get_symbol_direction("AAPL"), "long")
        self.assertEqual(bot.get_symbol_direction("TSLA"), "short")
        self.assertIn("AAPL", bot.get_scan_symbol_set())
        self.assertIn("TSLA", bot.get_scan_symbol_set())
        self.assertIn("AAPL", bot.get_monitored_extreme_symbols())
        self.assertIn("TSLA", bot.get_monitored_extreme_symbols())

    def test_human_focus_bounce_feedback_payload_is_tagged(self):
        bot = object.__new__(bounce_bot.BounceBot)
        payload = bot._build_bounce_feedback_alert_payload(
            "AAPL: Bounce confirmed (long) from ['ema_15']",
            {
                "symbol": "AAPL",
                "direction": "long",
                "bounce_types": "ema_15",
                "human_focus_pick": True,
                "human_focus_side": "LONG",
            },
        )

        self.assertTrue(payload["feedback"]["is_focus_pick"])
        self.assertEqual(payload["feedback"]["focus_side"], "LONG")

    def test_pending_bounce_does_not_finalize_before_eod_even_if_stop_or_target_seen(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outcomes_path = Path(temp_dir) / "intraday_bounce_outcomes.csv"
            bot = _test_bounce_bot_with_state(_pending_state(risk=1.0))
            df = _intraday_rows(
                (datetime(2026, 6, 1, 10, 5), 103.0, 98.0, 101.5),
                (datetime(2026, 6, 1, 10, 10), 102.0, 98.5, 100.5),
            )

            with (
                patch.object(bounce_bot, "INTRADAY_BOUNCE_OUTCOMES_CSV", outcomes_path),
                patch.object(
                    bounce_bot,
                    "get_market_local_now",
                    return_value=datetime(2026, 6, 1, 12, 0, tzinfo=timezone(timedelta(hours=-7))),
                ),
                patch.object(bounce_bot, "get_market_session_window", side_effect=_fake_session_window),
            ):
                bot._update_pending_bounce_outcomes("AAPL", df)

            self.assertIn("AAPL_long_1", bot.pending_bounce_outcomes)
            rows = pd.read_csv(outcomes_path)
            self.assertFalse((rows["event_type"] == "final").any())
            update_row = rows[rows["event_type"] == "update"].iloc[-1]
            self.assertEqual(update_row["status"], "open")
            self.assertTrue(bool(update_row["target_2r_hit"]))
            self.assertTrue(bool(update_row["stop_hit"]))

    def test_eod_finalization_uses_last_regular_session_bar_for_long_and_short(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outcomes_path = Path(temp_dir) / "intraday_bounce_outcomes.csv"
            after_close = datetime(2026, 6, 1, 13, 15, tzinfo=timezone(timedelta(hours=-7)))

            long_bot = _test_bounce_bot_with_state(_pending_state("AAPL_long_1", risk=2.0))
            long_df = _intraday_rows(
                (datetime(2026, 6, 1, 10, 5), 101.0, 99.0, 100.5),
                (datetime(2026, 6, 1, 12, 55), 107.0, 97.0, 106.0),
                (datetime(2026, 6, 1, 13, 5), 120.0, 119.0, 120.0),
            )

            short_bot = _test_bounce_bot_with_state(
                _pending_state("MSFT_short_1", symbol="MSFT", direction="short", risk=2.0)
            )
            short_df = _intraday_rows(
                (datetime(2026, 6, 1, 10, 5), 101.0, 99.0, 99.5),
                (datetime(2026, 6, 1, 12, 55), 103.0, 93.0, 94.0),
                (datetime(2026, 6, 1, 13, 5), 120.0, 119.0, 120.0),
            )

            with (
                patch.object(bounce_bot, "INTRADAY_BOUNCE_OUTCOMES_CSV", outcomes_path),
                patch.object(bounce_bot, "get_market_local_now", return_value=after_close),
                patch.object(bounce_bot, "get_market_session_window", side_effect=_fake_session_window),
            ):
                long_bot._update_pending_bounce_outcomes("AAPL", long_df)
                short_bot._update_pending_bounce_outcomes("MSFT", short_df)

            rows = pd.read_csv(outcomes_path)
            final_rows = rows[rows["event_type"] == "final"].set_index("event_id")

            long_final = final_rows.loc["AAPL_long_1"]
            self.assertEqual(long_final["status"], "eod_complete")
            self.assertAlmostEqual(long_final["eod_close"], 106.0)
            self.assertAlmostEqual(long_final["close_r"], 3.0)
            self.assertAlmostEqual(long_final["eod_move_pct"], 6.0)
            self.assertAlmostEqual(long_final["mfe_pct"], 7.0)
            self.assertAlmostEqual(long_final["mae_pct"], -3.0)

            short_final = final_rows.loc["MSFT_short_1"]
            self.assertEqual(short_final["status"], "eod_complete")
            self.assertAlmostEqual(short_final["eod_close"], 94.0)
            self.assertAlmostEqual(short_final["close_r"], 3.0)
            self.assertAlmostEqual(short_final["eod_move_pct"], 6.0)
            self.assertAlmostEqual(short_final["mfe_pct"], 7.0)
            self.assertAlmostEqual(short_final["mae_pct"], -3.0)

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
                    {
                        "event_id": "GOOG_long_old",
                        "event_type": "confirmed",
                        "logged_at": "2026-06-01T10:20:00",
                        "trade_date": "2026-06-01",
                        "symbol": "GOOG",
                        "direction": "long",
                        "bounce_types": "vwap",
                        "score": 40,
                        "risk_per_share": 0.35,
                        "rrs_spy": 1.0,
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
                        "eod_move_pct": 8.0,
                        "target_1r_hit": True,
                        "target_2r_hit": True,
                        "stop_hit": False,
                        "status": "eod_complete",
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
                        "eod_move_pct": 1.2,
                        "target_1r_hit": True,
                        "target_2r_hit": False,
                        "stop_hit": False,
                        "status": "eod_complete",
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
                        "status": "eod_complete",
                    },
                    {
                        "event_id": "GOOG_long_old",
                        "event_type": "final",
                        "logged_at": "2026-06-01T11:20:00",
                        "entry_time": "2026-06-01T10:20:00",
                        "bars_elapsed": 12,
                        "close_r": 3.0,
                        "mfe_r": 3.0,
                        "mae_r": -0.10,
                        "target_1r_hit": True,
                        "target_2r_hit": True,
                        "stop_hit": False,
                        "status": "target2_seen",
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
        self.assertEqual(vwap_row["recommendation"], "focus")
        self.assertIn("DT focus candidates", report_text)

    def test_intraday_bounce_performance_segments_top_pattern_h1_ema15_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            candidates_path = temp_path / "intraday_bounce_candidates.csv"
            outcomes_path = temp_path / "intraday_bounce_outcomes.csv"

            pd.DataFrame(
                [
                    {
                        "event_id": "TOPS_h1_1",
                        "event_type": "confirmed",
                        "logged_at": "2026-06-01T10:00:00",
                        "trade_date": "2026-06-01",
                        "symbol": "TOPS",
                        "direction": "long",
                        "bounce_types": "h1_ema_15",
                        "score": 42,
                        "risk_per_share": 0.50,
                        "rrs_spy": 1.0,
                        "market_environment": "bullish",
                        "master_avwap_setup_family": "top_pattern_tracking",
                        "master_avwap_h1_focus_type": "top_pattern",
                    },
                    {
                        "event_id": "TOPX_h1_2",
                        "event_type": "confirmed",
                        "logged_at": "2026-06-01T11:00:00",
                        "trade_date": "2026-06-01",
                        "symbol": "TOPX",
                        "direction": "long",
                        "bounce_types": "h1_ema_15",
                        "score": 43,
                        "risk_per_share": 0.55,
                        "rrs_spy": 1.2,
                        "market_environment": "bullish",
                        "master_avwap_setup_family": "top_pattern",
                        "master_avwap_h1_focus_type": "top_pattern",
                    },
                ]
            ).to_csv(candidates_path, index=False)
            pd.DataFrame(
                [
                    {
                        "event_id": "TOPS_h1_1",
                        "event_type": "final",
                        "logged_at": "2026-06-01T13:10:00",
                        "entry_time": "2026-06-01T10:00:00",
                        "bars_elapsed": 24,
                        "close_r": 0.70,
                        "mfe_r": 1.1,
                        "mae_r": -0.2,
                        "target_1r_hit": True,
                        "target_2r_hit": False,
                        "stop_hit": False,
                        "status": "eod_complete",
                    },
                    {
                        "event_id": "TOPX_h1_2",
                        "event_type": "final",
                        "logged_at": "2026-06-01T13:10:00",
                        "entry_time": "2026-06-01T11:00:00",
                        "bars_elapsed": 18,
                        "close_r": 0.50,
                        "mfe_r": 0.9,
                        "mae_r": -0.1,
                        "target_1r_hit": False,
                        "target_2r_hit": False,
                        "stop_hit": False,
                        "status": "eod_complete",
                    },
                ]
            ).to_csv(outcomes_path, index=False)

            rows = bounce_bot.build_intraday_bounce_performance_rows(
                candidates_path=candidates_path,
                outcomes_path=outcomes_path,
                min_samples=2,
            )

        h1_focus_row = next(
            row for row in rows
            if row["dimension"] == "master_avwap_h1_focus_type"
            and row["segment"] == "top_pattern"
            and row["direction"] == "long"
        )
        timing_row = next(
            row for row in rows
            if row["dimension"] == "top_pattern_entry_timing"
            and row["segment"] == "h1_15ema_bounce"
            and row["direction"] == "long"
        )
        self.assertEqual(h1_focus_row["sample_count"], 2)
        self.assertEqual(timing_row["sample_count"], 2)
        self.assertEqual(timing_row["recommendation"], "focus")

    def test_intraday_bounce_performance_segments_master_avwap_swing_traits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            candidates_path = temp_path / "intraday_bounce_candidates.csv"
            outcomes_path = temp_path / "intraday_bounce_outcomes.csv"

            pd.DataFrame(
                [
                    {
                        "event_id": "SWING1_long_1",
                        "event_type": "confirmed",
                        "logged_at": "2026-06-01T10:00:00",
                        "trade_date": "2026-06-01",
                        "symbol": "SWING1",
                        "direction": "long",
                        "bounce_types": "vwap",
                        "score": 42,
                        "risk_per_share": 0.50,
                        "rrs_spy": 1.0,
                        "market_environment": "bullish",
                        "master_avwap_swing_traits": "preferred_swing_focus;top_pattern_entry",
                    },
                    {
                        "event_id": "SWING2_long_1",
                        "event_type": "confirmed",
                        "logged_at": "2026-06-01T11:00:00",
                        "trade_date": "2026-06-01",
                        "symbol": "SWING2",
                        "direction": "long",
                        "bounce_types": "ema_21",
                        "score": 43,
                        "risk_per_share": 0.55,
                        "rrs_spy": 1.2,
                        "market_environment": "bullish",
                        "master_avwap_swing_traits": "preferred_swing_focus;top_pattern_entry",
                    },
                ]
            ).to_csv(candidates_path, index=False)
            pd.DataFrame(
                [
                    {
                        "event_id": "SWING1_long_1",
                        "event_type": "final",
                        "logged_at": "2026-06-01T13:10:00",
                        "entry_time": "2026-06-01T10:00:00",
                        "bars_elapsed": 24,
                        "close_r": 0.80,
                        "mfe_r": 1.4,
                        "mae_r": -0.1,
                        "target_1r_hit": True,
                        "target_2r_hit": False,
                        "stop_hit": False,
                        "status": "eod_complete",
                    },
                    {
                        "event_id": "SWING2_long_1",
                        "event_type": "final",
                        "logged_at": "2026-06-01T13:10:00",
                        "entry_time": "2026-06-01T11:00:00",
                        "bars_elapsed": 18,
                        "close_r": 0.60,
                        "mfe_r": 1.0,
                        "mae_r": -0.2,
                        "target_1r_hit": True,
                        "target_2r_hit": False,
                        "stop_hit": False,
                        "status": "eod_complete",
                    },
                ]
            ).to_csv(outcomes_path, index=False)

            rows = bounce_bot.build_intraday_bounce_performance_rows(
                candidates_path=candidates_path,
                outcomes_path=outcomes_path,
                min_samples=2,
            )

        trait_row = next(
            row for row in rows
            if row["dimension"] == "master_avwap_swing_trait"
            and row["segment"] == "top_pattern_entry"
            and row["direction"] == "long"
        )
        self.assertEqual(trait_row["sample_count"], 2)
        self.assertAlmostEqual(trait_row["avg_eod_r"], 0.70)
        self.assertAlmostEqual(trait_row["avg_mfe_r"], 1.20)
        self.assertEqual(trait_row["recommendation"], "focus")

    def test_intraday_bounce_performance_recommendation_categories_with_old_optional_columns_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            candidates_path = temp_path / "intraday_bounce_candidates.csv"
            outcomes_path = temp_path / "intraday_bounce_outcomes.csv"

            samples = [
                ("FOCUS1", "vwap", 0.60),
                ("FOCUS2", "vwap", 0.40),
                ("AVOID1", "ema_8", -0.50),
                ("AVOID2", "ema_8", -0.10),
                ("NEUT1", "dynamic_vwap", 0.30),
                ("NEUT2", "dynamic_vwap", -0.10),
                ("WATCH1", "eod_vwap", 0.80),
            ]
            pd.DataFrame(
                [
                    {
                        "event_id": event_id,
                        "event_type": "confirmed",
                        "logged_at": "2026-06-01T10:00:00",
                        "trade_date": "2026-06-01",
                        "symbol": event_id,
                        "direction": "long",
                        "bounce_types": bounce_type,
                        "score": 40,
                        "risk_per_share": 0.50,
                        "rrs_spy": 1.0,
                        "market_environment": "bullish",
                    }
                    for event_id, bounce_type, _close_r in samples
                ]
            ).to_csv(candidates_path, index=False)
            pd.DataFrame(
                [
                    {
                        "event_id": event_id,
                        "event_type": "final",
                        "logged_at": "2026-06-01T13:10:00",
                        "entry_time": "2026-06-01T10:00:00",
                        "bars_elapsed": 36,
                        "close_r": close_r,
                        "mfe_r": max(close_r, 0.0) + 0.20,
                        "mae_r": min(close_r, 0.0) - 0.10,
                        "target_1r_hit": close_r >= 1.0,
                        "target_2r_hit": close_r >= 2.0,
                        "stop_hit": close_r <= -1.0,
                        "status": "eod_complete",
                    }
                    for event_id, _bounce_type, close_r in samples
                ]
            ).to_csv(outcomes_path, index=False)

            rows = bounce_bot.build_intraday_bounce_performance_rows(
                candidates_path=candidates_path,
                outcomes_path=outcomes_path,
                min_samples=2,
            )

        by_type = {
            row["segment"]: row
            for row in rows
            if row["dimension"] == "bounce_type" and row["direction"] == "long"
        }
        self.assertEqual(by_type["vwap"]["recommendation"], "focus")
        self.assertEqual(by_type["ema_8"]["recommendation"], "avoid")
        self.assertEqual(by_type["dynamic_vwap"]["recommendation"], "neutral")
        self.assertEqual(by_type["eod_vwap"]["recommendation"], "watch_more_samples")

if __name__ == "__main__":
    unittest.main()
