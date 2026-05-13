import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_shared import (  # noqa: E402
    build_master_avwap_active_level_map,
    build_master_avwap_d1_flag_events,
    build_master_avwap_second_stdev_cross_map,
    describe_master_avwap_focus,
    load_master_avwap_d1_watchlist,
    load_master_avwap_events_for_date,
    load_master_avwap_focus_map,
    load_tradingview_groups,
)


class MasterAvwapSharedTests(unittest.TestCase):
    def test_event_loading_and_maps(self):
        trade_date = date.today()
        prior_date = trade_date - timedelta(days=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            signals_path = Path(temp_dir) / "avwap_signals.csv"
            signals_path.write_text(
                "\n".join(
                    [
                        "run_date,symbol,trade_date,side,anchor_type,anchor_date,signal_type,priority_bucket,favorite_zone,favorite_signals,is_favorite_setup,is_near_favorite_zone",
                        f"{trade_date.isoformat()},AAPL,{trade_date.isoformat()},LONG,CURRENT,2026-04-01,CROSS_UP_UPPER_2,favorite_setup,upper_band,CROSS_UP_UPPER_2,1,0",
                        f"{trade_date.isoformat()},TSLA,{trade_date.isoformat()},SHORT,PREVIOUS,2026-03-12,BOUNCE_LOWER_1,near_favorite_zone,lower_band,BOUNCE_LOWER_1,0,1",
                        f"{prior_date.isoformat()},MSFT,{prior_date.isoformat()},LONG,CURRENT,2026-03-20,CROSS_UP_VWAP,focus,,CROSS_UP_VWAP,0,0",
                        f"{trade_date.isoformat()},NVDA,{trade_date.isoformat()},LONG,CURRENT,2026-04-15,BETWEEN_AVWAP_AND_UPPER_1,focus,,,0,0",
                    ]
                ),
                encoding="utf-8",
            )

            events = load_master_avwap_events_for_date(
                trade_date=trade_date,
                signals_path=signals_path,
            )
            self.assertEqual(set(events.keys()), {"AAPL", "TSLA"})
            self.assertEqual(events["AAPL"][0]["level"], "UPPER_2")
            self.assertEqual(events["TSLA"][0]["level"], "LOWER_1")

            active_levels = build_master_avwap_active_level_map(events)
            self.assertEqual(active_levels["AAPL"], ["UPPER_2"])
            self.assertEqual(active_levels["TSLA"], ["LOWER_1"])

            second_stdev = build_master_avwap_second_stdev_cross_map(events)
            self.assertEqual(second_stdev["AAPL"]["signal_type"], "CROSS_UP_UPPER_2")
            self.assertEqual(second_stdev["AAPL"]["side"], "LONG")
            self.assertNotIn("TSLA", second_stdev)

    def test_focus_map_and_focus_feed_groups(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            focus_path = Path(temp_dir) / "master_avwap_focus.json"
            tradingview_path = Path(temp_dir) / "master_avwap_tradingview.txt"
            focus_path.write_text(
                json.dumps(
                    {
                        "symbols": {
                            "AAPL": {
                                "side": "LONG",
                                "priority_bucket": "favorite_setup",
                                "priority_rank": 1,
                                "priority_score": 99,
                                "favorite_zone": "upper_band",
                                "favorite_signals": ["cross_up_upper_2"],
                                "favorite_context_signals": ["prev_cross_up_vwap"],
                            },
                            "TSLA": {
                                "side": "SHORT",
                                "priority_bucket": "near_favorite_zone",
                                "favorite_zone": "lower_band",
                            },
                        },
                        "favorites": [{"symbol": "AAPL"}],
                        "near_favorite_zones": [{"symbol": "TSLA"}],
                    }
                ),
                encoding="utf-8",
            )

            focus_map = load_master_avwap_focus_map(focus_path=focus_path)
            self.assertEqual(focus_map["AAPL"]["favorite_signals"], ["CROSS_UP_UPPER_2"])
            self.assertEqual(
                focus_map["AAPL"]["favorite_context_signals"],
                ["PREV_CROSS_UP_VWAP"],
            )
            self.assertEqual(
                describe_master_avwap_focus(focus_map["AAPL"]),
                "best current favorite setup",
            )

            groups = load_tradingview_groups(
                focus_path=focus_path,
                tradingview_path=tradingview_path,
            )
            self.assertEqual(groups["source"], "focus_feed")
            self.assertEqual(groups["favorites"]["LONG"], ["AAPL"])
            self.assertEqual(groups["near_favorite_zones"]["SHORT"], ["TSLA"])

    def test_tradingview_report_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            focus_path = Path(temp_dir) / "missing_focus.json"
            tradingview_path = Path(temp_dir) / "master_avwap_tradingview.txt"
            tradingview_path.write_text(
                "\n".join(
                    [
                        "Best current favorite setups",
                        "LONG: NVDA, AMD",
                        "SHORT: TSLA",
                        "",
                        "Near favorite zones",
                        "LONG: MSFT",
                        "SHORT: NONE",
                    ]
                ),
                encoding="utf-8",
            )

            groups = load_tradingview_groups(
                focus_path=focus_path,
                tradingview_path=tradingview_path,
            )
            self.assertEqual(groups["source"], "tradingview_report")
            self.assertEqual(groups["favorites"]["LONG"], ["NVDA", "AMD"])
            self.assertEqual(groups["favorites"]["SHORT"], ["TSLA"])
            self.assertEqual(groups["near_favorite_zones"]["LONG"], ["MSFT"])
            self.assertEqual(groups["near_favorite_zones"]["SHORT"], [])

    def test_tradingview_report_preferred_over_stale_focus_feed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            focus_path = Path(temp_dir) / "master_avwap_focus.json"
            tradingview_path = Path(temp_dir) / "master_avwap_tradingview.txt"
            focus_path.write_text(
                json.dumps(
                    {
                        "symbols": {"OLD": {"side": "LONG"}},
                        "favorites": [{"symbol": "OLD", "side": "LONG"}],
                        "near_favorite_zones": [],
                    }
                ),
                encoding="utf-8",
            )
            tradingview_path.write_text(
                "\n".join(
                    [
                        "Master AVWAP TradingView lists",
                        "Generated at 2026-05-06 10:00:00",
                        "",
                        "Best current favorite setups",
                        "----------------------------",
                        "LONG: AAPL",
                        "SHORT: NONE",
                        "",
                        "Near favorite zones",
                        "-------------------",
                        "LONG: NVDA",
                        "SHORT: TSLA",
                    ]
                ),
                encoding="utf-8",
            )

            groups = load_tradingview_groups(
                focus_path=focus_path,
                tradingview_path=tradingview_path,
            )

            self.assertEqual(groups["source"], "tradingview_report")
            self.assertEqual(groups["favorites"]["LONG"], ["AAPL"])
            self.assertEqual(groups["near_favorite_zones"]["LONG"], ["NVDA"])
            self.assertEqual(groups["near_favorite_zones"]["SHORT"], ["TSLA"])

    def test_d1_watchlist_and_flag_events(self):
        trade_date = date.today()

        with tempfile.TemporaryDirectory() as temp_dir:
            watchlist_path = Path(temp_dir) / "master_avwap_d1_watchlist.json"
            watchlist_path.write_text(
                json.dumps(
                    {
                        "symbols": {
                            "NVDA": {
                                "side": "LONG",
                                "last_seen": trade_date.isoformat(),
                                "active_current_scan": True,
                                "priority_score": 260,
                                "watch_reasons": ["favorite_setup", "sold_put_premium_viable"],
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
                                "theta": {
                                    "play_type": "sold_put",
                                    "status": "recommended",
                                    "credit": 1.05,
                                    "strike": 190,
                                    "expiration": "20260515",
                                },
                            },
                            "APLD": {
                                "side": "LONG",
                                "last_seen": trade_date.isoformat(),
                                "priority_score": 190,
                                "watch_reasons": ["sold_put_premium_cusp"],
                                "theta": {
                                    "play_type": "sold_put",
                                    "status": "cusp",
                                    "credit": 0.18,
                                    "strike": 33,
                                },
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            d1_watchlist = load_master_avwap_d1_watchlist(watchlist_path=watchlist_path)
            self.assertEqual(d1_watchlist["NVDA"]["theta"]["status"], "recommended")
            self.assertEqual(d1_watchlist["NVDA"]["trigger_levels"][0]["level"], 102.0)
            self.assertEqual(d1_watchlist["NVDA"]["trigger_levels"][0]["action"], "break_above")

            focus_map = {
                "AAPL": {
                    "symbol": "AAPL",
                    "side": "LONG",
                    "priority_score": 240,
                    "mid_earnings_ema15_trigger": True,
                    "mid_earnings_note": "held 15EMA after earnings drift",
                },
                "TSLA": {
                    "symbol": "TSLA",
                    "side": "SHORT",
                    "priority_score": 210,
                    "trendline_break_recent": True,
                    "trendline_break_note": "broke rising support line",
                },
            }
            events_by_symbol = {
                "AAPL": [
                    {
                        "symbol": "AAPL",
                        "trade_date": trade_date,
                        "signal_type": "CROSS_UP_VWAP",
                        "anchor_type": "CURRENT",
                        "anchor_date": "2026-04-01",
                        "side": "LONG",
                        "level": "VWAP",
                    }
                ],
                "TSLA": [
                    {
                        "symbol": "TSLA",
                        "trade_date": trade_date,
                        "signal_type": "BOUNCE_LOWER_1",
                        "anchor_type": "CURRENT",
                        "anchor_date": "2026-04-02",
                        "side": "SHORT",
                        "level": "LOWER_1",
                    }
                ],
            }

            flags = build_master_avwap_d1_flag_events(focus_map, events_by_symbol, d1_watchlist, trade_date)
            labels_by_symbol = {}
            for event in flags:
                labels_by_symbol.setdefault(event["symbol"], set()).add(event["label"])

            self.assertIn("AVWAPE breakthrough", labels_by_symbol["AAPL"])
            self.assertIn("15EMA D1 bounce", labels_by_symbol["AAPL"])
            self.assertIn("1st-dev D1 bounce", labels_by_symbol["TSLA"])
            self.assertIn("Trendline breakthrough", labels_by_symbol["TSLA"])
            self.assertIn("Put premium viable", labels_by_symbol["NVDA"])
            self.assertIn("Put premium cusp", labels_by_symbol["APLD"])


if __name__ == "__main__":
    unittest.main()
