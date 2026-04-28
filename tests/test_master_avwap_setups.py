import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap  # noqa: E402
from master_avwap import (  # noqa: E402
    POST_EARNINGS_STOP_FAILURE_CLOSES,
    POST_EARNINGS_STOP_LABEL,
    _build_latest_earnings_release_context,
    _evaluate_tracker_scenario_bar,
    _find_tracker_stop_candidates,
    analyze_mid_earnings_ema_retest_setup,
    append_d1_feature_history,
    append_master_avwap_user_favorites,
    apply_tracker_setup_type_adjustments,
    apply_recent_tracker_setup_family_adjustments,
    attach_setup_candidate_payloads,
    build_market_prep_payload,
    build_priority_setup_summary,
    build_master_avwap_focus_setup_type_text,
    build_master_avwap_focus_side_groups,
    build_recent_tracker_setup_family_rows,
    format_market_prep_payload_report,
    load_scan_earnings_context,
    rank_tracker_setup_type_rows,
    write_stdev_range_report,
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


def _build_mid_earnings_history() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=25)
    rows = []
    for idx, dt_value in enumerate(dates):
        close_price = 110.0
        if idx >= len(dates) - 2:
            close_price = 121.5 + (idx - (len(dates) - 2))
        rows.append(
            {
                "datetime": dt_value,
                "open": close_price - 1.0,
                "high": close_price + 1.0,
                "low": close_price - 2.0,
                "close": close_price,
                "volume": 1_500_000,
            }
        )
    return pd.DataFrame(rows)


def _build_mid_earnings_retest_history(side: str = "LONG") -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=25)
    rows = []
    for idx, dt_value in enumerate(dates):
        if side == "LONG":
            close_price = 110.0
            if idx == 20:
                close_price = 121.5
            elif idx == 21:
                close_price = 122.5
            elif idx == 22:
                close_price = 118.0
            elif idx == 23:
                close_price = 116.5
            elif idx == 24:
                close_price = 117.0
        else:
            close_price = 110.0
            if idx == 20:
                close_price = 98.5
            elif idx == 21:
                close_price = 97.5
            elif idx == 22:
                close_price = 102.0
            elif idx == 23:
                close_price = 103.5
            elif idx == 24:
                close_price = 104.0
        rows.append(
            {
                "datetime": dt_value,
                "open": close_price - 1.0,
                "high": close_price + 1.0,
                "low": close_price - 2.0,
                "close": close_price,
                "volume": 1_500_000,
            }
        )
    return pd.DataFrame(rows)


def _build_mid_earnings_band_history(df: pd.DataFrame) -> dict:
    bands = {
        "UPPER_1": 115.0,
        "UPPER_2": 120.0,
        "UPPER_3": 126.0,
        "LOWER_1": 105.0,
        "LOWER_2": 100.0,
        "LOWER_3": 95.0,
    }
    return {
        row["datetime"].date().isoformat(): {"bands": dict(bands)}
        for _, row in df.iterrows()
    }


def _build_mid_earnings_release_context(
    df: pd.DataFrame,
    *,
    sessions_since_gap: int,
) -> dict:
    anchor_date = df.iloc[3]["datetime"].date().isoformat()
    gap_date = df.iloc[6]["datetime"].date().isoformat()
    bands = {
        "UPPER_1": 115.0,
        "UPPER_2": 120.0,
        "UPPER_3": 126.0,
        "LOWER_1": 105.0,
        "LOWER_2": 100.0,
        "LOWER_3": 95.0,
    }
    return {
        "active": True,
        "anchor_date": anchor_date,
        "gap_date": gap_date,
        "sessions_since_gap": sessions_since_gap,
        "anchor_meta": {
            "date": anchor_date,
            "bands": dict(bands),
        },
    }


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


def _build_setup_type_stat(
    *,
    side: str,
    favorite_zone: str,
    retest_label: str,
    compression_label: str = "N",
    priority_bucket: str = "favorite_setup",
    setup_family: str = "general",
    tracked_setups: int = 20,
    closed_setups: int = 12,
    avg_closed_r: float = 0.0,
    baseline_avg_closed_r: float = 0.0,
    avg_total_r: float | None = None,
    baseline_avg_total_r: float | None = None,
    target_hit_rate: float = 0.5,
    baseline_target_hit_rate: float = 0.5,
    stop_rate: float = 0.5,
    baseline_stop_rate: float = 0.5,
) -> dict:
    avg_total = avg_closed_r if avg_total_r is None else avg_total_r
    baseline_total = baseline_avg_closed_r if baseline_avg_total_r is None else baseline_avg_total_r
    return {
        "setup_type_id": " | ".join(
            [side, priority_bucket, setup_family, favorite_zone, retest_label, compression_label]
        ),
        "type_label": (
            f"{side} | {priority_bucket} | family={setup_family} | zone={favorite_zone} "
            f"| retest={retest_label} | comp={compression_label}"
        ),
        "side": side,
        "priority_bucket": priority_bucket,
        "setup_family": setup_family,
        "favorite_zone": favorite_zone,
        "retest_label": retest_label,
        "compression_label": compression_label,
        "tracked_setups": tracked_setups,
        "tradeable_setups": tracked_setups,
        "closed_setups": closed_setups,
        "open_setups": max(0, tracked_setups - closed_setups),
        "avg_priority_score": 100.0,
        "avg_total_r": avg_total,
        "avg_closed_r": avg_closed_r,
        "target_hit_rate": target_hit_rate,
        "stop_rate": stop_rate,
        "baseline_avg_total_r": baseline_total,
        "baseline_avg_closed_r": baseline_avg_closed_r,
        "baseline_target_hit_rate": baseline_target_hit_rate,
        "baseline_stop_rate": baseline_stop_rate,
        "avg_total_r_edge": avg_total - baseline_total,
        "avg_closed_r_edge": avg_closed_r - baseline_avg_closed_r,
        "target_hit_rate_edge": target_hit_rate - baseline_target_hit_rate,
        "stop_rate_edge": stop_rate - baseline_stop_rate,
        "sample_setups": "",
    }


class MasterAvwapSetupTests(unittest.TestCase):
    def test_mid_earnings_hold_requires_15_sessions_after_gap(self):
        df = _build_mid_earnings_history()
        anchor_date = df.iloc[3]["datetime"].date().isoformat()
        gap_date = df.iloc[6]["datetime"].date().isoformat()
        band_history = {
            row["datetime"].date().isoformat(): {
                "bands": {
                    "UPPER_1": 115.0,
                    "UPPER_2": 120.0,
                    "UPPER_3": 126.0,
                    "LOWER_1": 105.0,
                    "LOWER_2": 100.0,
                    "LOWER_3": 95.0,
                }
            }
            for _, row in df.iterrows()
        }
        release_context = {
            "active": True,
            "anchor_date": anchor_date,
            "gap_date": gap_date,
            "sessions_since_gap": 14,
            "anchor_meta": {
                "date": anchor_date,
                "bands": {
                    "UPPER_1": 115.0,
                    "UPPER_2": 120.0,
                    "UPPER_3": 126.0,
                    "LOWER_1": 105.0,
                    "LOWER_2": 100.0,
                    "LOWER_3": 95.0,
                },
            },
        }

        with patch("master_avwap.calc_anchored_vwap_band_history", return_value=band_history):
            result = analyze_mid_earnings_ema_retest_setup(df, "LONG", release_context, indicator_frame=None)

        self.assertFalse(result["watch"])
        self.assertFalse(result["favorite_signal"])
        self.assertFalse(result["active_second_stdev_hold"])

    def test_mid_earnings_hold_detects_current_above_second_stdev_streak(self):
        df = _build_mid_earnings_history()
        anchor_date = df.iloc[3]["datetime"].date().isoformat()
        gap_date = df.iloc[6]["datetime"].date().isoformat()
        band_history = {
            row["datetime"].date().isoformat(): {
                "bands": {
                    "UPPER_1": 115.0,
                    "UPPER_2": 120.0,
                    "UPPER_3": 126.0,
                    "LOWER_1": 105.0,
                    "LOWER_2": 100.0,
                    "LOWER_3": 95.0,
                }
            }
            for _, row in df.iterrows()
        }
        release_context = {
            "active": True,
            "anchor_date": anchor_date,
            "gap_date": gap_date,
            "sessions_since_gap": 16,
            "anchor_meta": {
                "date": anchor_date,
                "bands": {
                    "UPPER_1": 115.0,
                    "UPPER_2": 120.0,
                    "UPPER_3": 126.0,
                    "LOWER_1": 105.0,
                    "LOWER_2": 100.0,
                    "LOWER_3": 95.0,
                },
            },
        }

        with patch("master_avwap.calc_anchored_vwap_band_history", return_value=band_history):
            result = analyze_mid_earnings_ema_retest_setup(df, "LONG", release_context, indicator_frame=None)

        self.assertTrue(result["watch"])
        self.assertFalse(result["favorite_signal"])
        self.assertTrue(result["active_second_stdev_hold"])
        self.assertEqual(result["zone_streak_days"], 2)
        self.assertIn("16 sessions since post-earnings gap", result["note"])
        self.assertIn("still trading above UPPER_2", result["note"])

    def test_mid_earnings_active_hold_gets_distinct_setup_family(self):
        summary = build_priority_setup_summary(
            symbol="AAPL",
            side="LONG",
            events_today=[],
            all_events=[],
            trend_label="UP",
            favorite_zone=None,
            mid_earnings_active_second_stdev_hold=True,
        )

        self.assertEqual(summary["setup_family"], "mid_earnings_above_2nd_stdev")

    def test_mid_earnings_ema21_retest_sets_family_without_double_counting_confluence(self):
        df = _build_mid_earnings_retest_history("LONG")
        release_context = _build_mid_earnings_release_context(df, sessions_since_gap=16)
        release_context["anchor_meta"]["bands"]["UPPER_1"] = 112.0
        band_history = _build_mid_earnings_band_history(df)
        for state in band_history.values():
            state["bands"]["UPPER_1"] = 112.0
        indicator_frame = pd.DataFrame(
            [
                {
                    "trade_date": df.iloc[-1]["datetime"].date().isoformat(),
                    "ema_8": 119.0,
                    "ema_15": 118.0,
                    "ema_21": 116.0,
                }
            ]
        )

        with patch(
            "master_avwap.calc_anchored_vwap_band_history",
            return_value=band_history,
        ):
            result = analyze_mid_earnings_ema_retest_setup(
                df,
                "LONG",
                release_context,
                indicator_frame=indicator_frame,
            )

        self.assertTrue(result["favorite_signal"])
        self.assertEqual(result["primary_trigger_level"], "EMA_21")
        self.assertEqual(result["trigger_levels"], ["EMA_21"])
        self.assertTrue(result["ema21_trigger"])
        self.assertFalse(result["ema21_confluence"])
        self.assertIn("MID_EARNINGS_EMA21_RETEST", result["events"])
        self.assertNotIn("MID_EARNINGS_EMA21_CONFLUENCE", result["events"])

        summary = build_priority_setup_summary(
            symbol="AAPL",
            side="LONG",
            events_today=result["events"],
            all_events=result["events"],
            trend_label="UP",
            favorite_zone=None,
            mid_earnings_primary_trigger_level=result["primary_trigger_level"],
        )

        self.assertEqual(summary["setup_family"], "mid_earnings_ema21_retest")

    def test_mid_earnings_recent_second_dev_hold_waits_for_actual_retest(self):
        df = _build_mid_earnings_retest_history("LONG")
        for idx in (22, 23, 24):
            df.loc[idx, "open"] = 118.2
            df.loc[idx, "high"] = 119.0
            df.loc[idx, "low"] = 117.7
            df.loc[idx, "close"] = 118.4
        release_context = _build_mid_earnings_release_context(df, sessions_since_gap=18)
        indicator_frame = pd.DataFrame(
            [
                {
                    "trade_date": df.iloc[-1]["datetime"].date().isoformat(),
                    "ema_8": 119.0,
                    "ema_15": 116.0,
                    "ema_21": 115.0,
                }
            ]
        )

        with patch(
            "master_avwap.calc_anchored_vwap_band_history",
            return_value=_build_mid_earnings_band_history(df),
        ):
            result = analyze_mid_earnings_ema_retest_setup(
                df,
                "LONG",
                release_context,
                indicator_frame=indicator_frame,
            )

        self.assertTrue(result["watch"])
        self.assertFalse(result["favorite_signal"])
        self.assertEqual(result["zone_streak_days"], 2)
        self.assertEqual(result["sessions_after_zone"], 3)
        self.assertIn("waiting for a quick EMA15 / EMA21 / UPPER_1 retest", result["note"])

    def test_mid_earnings_first_dev_short_trigger_uses_post_earnings_anchor_stop(self):
        df = _build_mid_earnings_retest_history("SHORT")
        release_context = _build_mid_earnings_release_context(df, sessions_since_gap=17)
        indicator_frame = pd.DataFrame(
            [
                {
                    "trade_date": df.iloc[-1]["datetime"].date().isoformat(),
                    "ema_8": 109.0,
                    "ema_15": 108.0,
                    "ema_21": 106.0,
                }
            ]
        )

        with patch(
            "master_avwap.calc_anchored_vwap_band_history",
            return_value=_build_mid_earnings_band_history(df),
        ):
            result = analyze_mid_earnings_ema_retest_setup(
                df,
                "SHORT",
                release_context,
                indicator_frame=indicator_frame,
            )

        self.assertTrue(result["favorite_signal"])
        self.assertEqual(result["primary_trigger_level"], "LOWER_1")
        self.assertIn("MID_EARNINGS_FIRST_DEV_RETEST", result["events"])
        self.assertTrue(result["first_dev_trigger"])

        summary = build_priority_setup_summary(
            symbol="TSLA",
            side="SHORT",
            events_today=result["events"],
            all_events=result["events"],
            trend_label="DOWN",
            favorite_zone=None,
            mid_earnings_primary_trigger_level=result["primary_trigger_level"],
        )

        self.assertEqual(summary["setup_family"], "mid_earnings_1stdev_retest")

        candidates = _find_tracker_stop_candidates(
            {
                "side": "SHORT",
                "priority_bucket": "favorite_setup",
                "setup_family": "mid_earnings_1stdev_retest",
            },
            {
                "side": "SHORT",
                "current_anchor": {
                    "vwap": 100.0,
                    "bands": {"UPPER_1": 120.0, "LOWER_1": 99.0},
                },
                "post_earnings_anchor": {
                    "vwap": 101.0,
                    "bands": {"LOWER_1": 105.0},
                },
                "ema_15": 108.0,
                "ema_21": 106.0,
                "ema_8": 109.0,
                "atr20": 2.0,
            },
        )

        first_dev_stop = next(item for item in candidates if item["label"] == "LOWER_1")
        self.assertEqual(first_dev_stop["source_type"], "post_earnings_anchor")
        self.assertEqual(first_dev_stop["level"], 105.0)

    def test_stdev_report_includes_mid_earnings_above_second_stdev_section(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "stdev_report.txt"
            write_stdev_range_report(
                report_path,
                {"long": [], "short": []},
                {"long": [], "short": []},
                priority_rows=[
                    {
                        "symbol": "AAPL",
                        "side": "LONG",
                        "score": 101,
                        "setup_family": "mid_earnings_above_2nd_stdev",
                    },
                    {
                        "symbol": "TSLA",
                        "side": "SHORT",
                        "score": 99,
                        "setup_family": "mid_earnings_above_2nd_stdev",
                    },
                ],
            )

            text = report_path.read_text(encoding="utf-8")

        self.assertIn("15+ sessions after earnings and above 2nd stdev for >= 2 days", text)
        self.assertIn("Longs (above UPPER_2): AAPL", text)
        self.assertIn("Shorts (below LOWER_2): TSLA", text)

    def test_focus_side_groups_split_candidates_by_direction(self):
        payload = {
            "favorites": [
                {"symbol": "AAPL", "side": "LONG", "priority_score": 99, "setup_family": "avwap_breakout"},
                {"symbol": "TSLA", "side": "SHORT", "priority_score": 97, "setup_family": "avwap_breakdown"},
            ],
            "near_favorite_zones": [
                {"symbol": "MSFT", "side": "LONG", "priority_score": 95, "setup_family": "avwap_breakout"},
                {"symbol": "AAPL", "side": "LONG", "priority_score": 80, "setup_family": "duplicate_should_skip"},
            ],
            "post_earnings_plays": [
                {"symbol": "NVDA", "side": "LONG", "priority_score": 94, "setup_family": "post_earnings_52w_break"},
                {"symbol": "TSLA", "side": "SHORT", "priority_score": 70, "setup_family": "duplicate_should_skip"},
            ],
        }

        groups = build_master_avwap_focus_side_groups(payload)

        self.assertEqual(groups["LONG"], ["AAPL", "MSFT", "NVDA"])
        self.assertEqual(groups["SHORT"], ["TSLA"])

    def test_focus_setup_type_text_groups_symbols_by_family(self):
        payload = {
            "favorites": [
                {"symbol": "AAPL", "side": "LONG", "priority_score": 99, "setup_family": "avwap_breakout"},
                {"symbol": "TSLA", "side": "SHORT", "priority_score": 97, "setup_family": "avwap_breakdown"},
            ],
            "near_favorite_zones": [
                {"symbol": "MSFT", "side": "LONG", "priority_score": 95, "setup_family": "avwap_breakout"},
            ],
            "post_earnings_plays": [
                {"symbol": "NVDA", "side": "LONG", "priority_score": 94, "setup_family": "post_earnings_52w_break"},
            ],
        }

        text = build_master_avwap_focus_setup_type_text(payload)

        self.assertEqual(
            text,
            "\n".join(
                [
                    "avwap_breakout",
                    "LONG: AAPL, MSFT",
                    "",
                    "avwap_breakdown",
                    "SHORT: TSLA",
                    "",
                    "post_earnings_52w_break",
                    "LONG: NVDA",
                ]
            ),
        )

    def test_market_prep_payload_builds_requested_copy_sections(self):
        payload = build_market_prep_payload(
            range_buckets={
                "long_avwap_to_upper_1": ["MSFT", "AAPL"],
                "short_avwap_to_lower_1": ["TSLA"],
            },
            market_prep_range_buckets={
                "long_upper_2_to_upper_3_2_sessions": ["NVDA"],
            },
            priority_rows=[
                {
                    "symbol": "AMD",
                    "side": "SHORT",
                    "score": 121.5,
                    "post_earnings_active": True,
                    "post_earnings_break_intraday": True,
                    "post_earnings_break_close": True,
                    "post_earnings_note": "new 52-week closing low; close confirmed",
                },
                {
                    "symbol": "SMCI",
                    "side": "LONG",
                    "score": 92,
                    "post_earnings_active": True,
                    "post_earnings_break_intraday": False,
                    "post_earnings_note": "qualified post-earnings high watch",
                },
                {
                    "symbol": "QQQ",
                    "side": "LONG",
                    "score": 200,
                    "post_earnings_active": False,
                },
            ],
            latest_release_map={
                "META": {"earnings_date": "2026-04-24", "release_session": "bmo"},
                "NFLX": {"earnings_date": "2026-04-23", "release_session": "amc"},
                "LATE": {"earnings_date": "2026-04-24", "release_session": "amc"},
            },
            calendar_rows_by_date={
                "2026-04-23": [
                    {"symbol": "NFLX", "time": "After Market Close"},
                ],
                "2026-04-24": [
                    {"symbol": "META", "time": "Before Market Open"},
                    {"symbol": "LATE", "time": "After Market Close"},
                ],
            },
            reference_date=date(2026, 4, 24),
            previous_session_date=date(2026, 4, 23),
        )

        sections = {section["id"]: section for section in payload["sections"]}

        self.assertEqual(sections["long_avwape_to_1stdev"]["copy_text"], "AAPL, MSFT")
        self.assertEqual(sections["short_avwape_to_1stdev"]["copy_text"], "TSLA")
        self.assertEqual(sections["long_2nd_to_3rd_stdev_2_sessions"]["copy_text"], "NVDA")
        self.assertEqual(sections["earnings_last_night_or_today"]["copy_text"], "META, NFLX")
        self.assertNotIn("LATE", sections["earnings_last_night_or_today"]["copy_text"])
        self.assertEqual(sections["post_earnings_potential_plays"]["copy_text"], "AMD")

        report = format_market_prep_payload_report(payload)

        self.assertIn("Longs: AVWAPE to 1st Dev", report)
        self.assertIn("NFLX   last_night earnings=2026-04-23", report)
        self.assertIn("AMD    SHORT close_confirmed score=121.5", report)

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

    def test_recent_tracker_family_open_only_winners_do_not_get_positive_boost(self):
        reference_date = date(2026, 4, 22)
        setups = {
            "open_1": _build_tracker_setup(
                "AAPL",
                "2026-04-21",
                2.1,
                setup_family="avwap_retest_followthrough",
                scenario_status="OPEN",
            ),
            "open_2": _build_tracker_setup(
                "NVDA",
                "2026-04-20",
                1.9,
                setup_family="avwap_retest_followthrough",
                scenario_status="OPEN",
            ),
            "open_3": _build_tracker_setup(
                "META",
                "2026-04-19",
                1.6,
                setup_family="avwap_retest_followthrough",
                scenario_status="OPEN",
            ),
            "baseline_1": _build_tracker_setup("MSFT", "2026-04-21", 0.1, setup_family="avwap_breakout", scenario_status="OPEN"),
            "baseline_2": _build_tracker_setup("CRM", "2026-04-20", 0.0, setup_family="avwap_breakout", scenario_status="OPEN"),
            "baseline_3": _build_tracker_setup("ORCL", "2026-04-19", 0.2, setup_family="avwap_breakout", scenario_status="OPEN"),
        }

        recent_rows = build_recent_tracker_setup_family_rows(setups, reference_date=reference_date)
        open_winner_row = next(
            row for row in recent_rows
            if row["setup_family"] == "avwap_retest_followthrough"
        )

        self.assertEqual(open_winner_row["closed_setups"], 0)
        self.assertEqual(open_winner_row["score_delta"], 0)

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

    def test_rank_tracker_setup_type_rows_prefers_best_context_within_side_bucket(self):
        setup_type_rows = [
            _build_setup_type_stat(
                side="LONG",
                favorite_zone="AVWAPE to UPPER_1",
                retest_label="None",
                tracked_setups=48,
                closed_setups=24,
                avg_closed_r=1.80,
                baseline_avg_closed_r=0.35,
                target_hit_rate=0.74,
                baseline_target_hit_rate=0.51,
                stop_rate=0.32,
                baseline_stop_rate=0.57,
            ),
            _build_setup_type_stat(
                side="LONG",
                favorite_zone="None",
                retest_label="AVWAPE",
                tracked_setups=42,
                closed_setups=20,
                avg_closed_r=0.55,
                baseline_avg_closed_r=0.35,
                target_hit_rate=0.58,
                baseline_target_hit_rate=0.51,
                stop_rate=0.49,
                baseline_stop_rate=0.57,
            ),
            _build_setup_type_stat(
                side="LONG",
                favorite_zone="None",
                retest_label="None",
                tracked_setups=38,
                closed_setups=18,
                avg_closed_r=-0.10,
                baseline_avg_closed_r=0.35,
                target_hit_rate=0.44,
                baseline_target_hit_rate=0.51,
                stop_rate=0.62,
                baseline_stop_rate=0.57,
            ),
            _build_setup_type_stat(
                side="SHORT",
                favorite_zone="None",
                retest_label="UPPER_1",
                tracked_setups=23,
                closed_setups=23,
                avg_closed_r=0.52,
                baseline_avg_closed_r=-0.35,
                target_hit_rate=1.00,
                baseline_target_hit_rate=0.62,
                stop_rate=1.00,
                baseline_stop_rate=1.00,
            ),
        ]

        ranked_rows = rank_tracker_setup_type_rows(setup_type_rows)
        ranked_lookup = {
            (row["side"], row["favorite_zone"], row["retest_label"]): row
            for row in ranked_rows
        }

        best_long = ranked_lookup[("LONG", "AVWAPE to UPPER_1", "None")]
        mid_long = ranked_lookup[("LONG", "None", "AVWAPE")]
        weak_long = ranked_lookup[("LONG", "None", "None")]
        best_short = ranked_lookup[("SHORT", "None", "UPPER_1")]

        self.assertEqual(best_long["rank_within_side_bucket"], 1)
        self.assertGreater(best_long["score_delta"], mid_long["score_delta"])
        self.assertGreater(mid_long["score_delta"], weak_long["score_delta"])
        self.assertEqual(best_short["rank_within_side_bucket"], 1)
        self.assertGreater(best_short["score_delta"], 0)

    def test_setup_type_adjustment_reorders_equal_base_scores_by_historical_context(self):
        tracker_payload = {
            "setup_type_stats": [
                _build_setup_type_stat(
                    side="LONG",
                    favorite_zone="AVWAPE to UPPER_1",
                    retest_label="None",
                    tracked_setups=48,
                    closed_setups=24,
                    avg_closed_r=1.80,
                    baseline_avg_closed_r=0.35,
                    target_hit_rate=0.74,
                    baseline_target_hit_rate=0.51,
                    stop_rate=0.32,
                    baseline_stop_rate=0.57,
                ),
                _build_setup_type_stat(
                    side="LONG",
                    favorite_zone="None",
                    retest_label="AVWAPE",
                    tracked_setups=42,
                    closed_setups=20,
                    avg_closed_r=0.55,
                    baseline_avg_closed_r=0.35,
                    target_hit_rate=0.58,
                    baseline_target_hit_rate=0.51,
                    stop_rate=0.49,
                    baseline_stop_rate=0.57,
                ),
            ]
        }
        priority_rows = [
            {
                "symbol": "AAPL",
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "setup_family": "general",
                "favorite_zone": "AVWAPE to UPPER_1",
                "retest_followthrough": False,
                "retest_reference_level": "",
                "compression_flag": False,
                "score": 100.0,
            },
            {
                "symbol": "MSFT",
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "setup_family": "general",
                "favorite_zone": "None",
                "retest_followthrough": True,
                "retest_reference_level": "AVWAPE",
                "compression_flag": False,
                "score": 100.0,
            },
        ]
        ai_state = {"symbols": {"AAPL": {}, "MSFT": {}}}
        feature_rows_by_symbol = {
            "AAPL": {"priority_score": 100.0},
            "MSFT": {"priority_score": 100.0},
        }

        apply_tracker_setup_type_adjustments(
            priority_rows,
            ai_state,
            feature_rows_by_symbol,
            tracker_payload=tracker_payload,
        )

        self.assertGreater(priority_rows[0]["score"], priority_rows[1]["score"])
        self.assertTrue(priority_rows[0]["setup_type_score_note"])
        self.assertEqual(feature_rows_by_symbol["AAPL"]["priority_score"], priority_rows[0]["score"])
        self.assertEqual(ai_state["symbols"]["AAPL"]["priority_score"], priority_rows[0]["score"])

    def test_setup_type_adjustment_uses_mid_earnings_primary_trigger_level(self):
        tracker_payload = {
            "setup_type_stats": [
                _build_setup_type_stat(
                    side="LONG",
                    favorite_zone="None",
                    retest_label="EMA_21",
                    setup_family="mid_earnings_ema21_retest",
                    tracked_setups=24,
                    closed_setups=14,
                    avg_closed_r=1.25,
                    baseline_avg_closed_r=0.15,
                    target_hit_rate=0.71,
                    baseline_target_hit_rate=0.48,
                    stop_rate=0.29,
                    baseline_stop_rate=0.56,
                ),
            ]
        }
        priority_rows = [
            {
                "symbol": "AAPL",
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "setup_family": "mid_earnings_ema21_retest",
                "favorite_zone": "None",
                "retest_followthrough": False,
                "retest_reference_level": "",
                "mid_earnings_primary_trigger_level": "EMA_21",
                "compression_flag": False,
                "score": 100.0,
            },
        ]
        ai_state = {"symbols": {"AAPL": {}}}
        feature_rows_by_symbol = {
            "AAPL": {"priority_score": 100.0},
        }

        apply_tracker_setup_type_adjustments(
            priority_rows,
            ai_state,
            feature_rows_by_symbol,
            tracker_payload=tracker_payload,
        )

        self.assertGreater(priority_rows[0]["score"], 100.0)
        self.assertIn("setup type rank", priority_rows[0]["setup_type_score_note"])
        self.assertEqual(feature_rows_by_symbol["AAPL"]["priority_score"], priority_rows[0]["score"])
        self.assertEqual(ai_state["symbols"]["AAPL"]["priority_score"], priority_rows[0]["score"])

    def test_market_regime_penalizes_countertrend_without_fresh_trigger(self):
        priority_rows = [
            {
                "symbol": "TSLA",
                "side": "SHORT",
                "trend_20d": "UP",
                "score": 100.0,
            }
        ]
        ai_state = {
            "market_regime": {"label": "bullish"},
            "symbols": {"TSLA": {"side": "SHORT", "trend_20d": "UP"}},
        }
        feature_rows_by_symbol = {"TSLA": {"priority_score": 100.0}}

        master_avwap.apply_market_regime_score_adjustments(
            priority_rows,
            ai_state,
            feature_rows_by_symbol,
        )

        self.assertEqual(priority_rows[0]["market_regime_score_delta"], -18)
        self.assertEqual(priority_rows[0]["score"], 82.0)
        self.assertEqual(feature_rows_by_symbol["TSLA"]["priority_score"], 82.0)

    def test_rejection_cap_skips_valid_mid_earnings_retest_second_band_history(self):
        priority_rows = [
            {
                "symbol": "AAOI",
                "side": "LONG",
                "setup_family": "mid_earnings_ema15_retest",
                "score": 449.0,
                "second_band_penalty": 22,
            },
            {
                "symbol": "CHOP",
                "side": "LONG",
                "setup_family": "avwap_breakout",
                "score": 310.0,
                "second_band_penalty": 22,
            },
        ]
        ai_state = {"symbols": {"AAOI": {}, "CHOP": {}}}
        feature_rows_by_symbol = {
            "AAOI": {"priority_score": 449.0},
            "CHOP": {"priority_score": 310.0},
        }

        master_avwap.apply_priority_rejection_score_caps(
            priority_rows,
            ai_state,
            feature_rows_by_symbol,
        )

        self.assertIsNone(priority_rows[0]["rejection_score_cap"])
        self.assertEqual(priority_rows[0]["score"], 449.0)
        self.assertEqual(priority_rows[1]["rejection_score_cap"], 240.0)
        self.assertEqual(priority_rows[1]["score"], 240.0)

    def test_attach_setup_candidate_payloads_records_common_candidate_shape(self):
        priority_rows = [
            {
                "symbol": "AAPL",
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "setup_family": "avwap_breakout",
                "setup_tags": ["breakout"],
                "favorite_zone": "AVWAPE to UPPER_1",
                "favorite_signals": ["CROSS_UP_UPPER_1"],
                "context_signals": [],
                "score": 120,
                "base_score": 100,
                "ranking_blocked": True,
                "ranking_block_reason": "SMA obstacle",
            }
        ]
        ai_state = {
            "symbols": {
                "AAPL": {
                    "side": "LONG",
                    "current_anchor": {
                        "vwap": 100.0,
                        "bands": {"UPPER_2": 112.0, "UPPER_3": 120.0},
                    },
                    "daily_bar_source": "ibkr",
                    "market_regime_label": "bullish",
                }
            }
        }
        feature_rows_by_symbol = {"AAPL": {"symbol": "AAPL"}}

        attach_setup_candidate_payloads(priority_rows, ai_state, feature_rows_by_symbol)

        candidate = priority_rows[0]["setup_candidate"]
        self.assertEqual(candidate["schema_version"], master_avwap.SETUP_CANDIDATE_SCHEMA_VERSION)
        self.assertEqual(candidate["family"], "avwap_breakout")
        self.assertEqual(candidate["targets"][0]["label"], "UPPER_2")
        self.assertIn("SMA obstacle", candidate["rejection_reasons"])
        self.assertIn("setup_candidate_json", feature_rows_by_symbol["AAPL"])
        self.assertEqual(ai_state["symbols"]["AAPL"]["setup_candidate"]["family"], "avwap_breakout")

    def test_append_d1_feature_history_adds_run_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "d1_features_history.csv"
            frame = pd.DataFrame(
                [
                    {
                        "symbol": "AAPL",
                        "side": "LONG",
                        "priority_score": 123.0,
                    }
                ]
            )
            metadata = {
                "run_id": "run-1",
                "run_timestamp": "2026-04-24T13:00:00",
                "run_date": "2026-04-24",
                "watchlist_label": "test-watchlist",
                "scoring_config_hash": "abc123",
                "scoring_config_updated_at": "2026-04-24T12:00:00",
            }
            with patch.object(master_avwap, "D1_FEATURE_HISTORY_FILE", history_path):
                append_d1_feature_history(frame, metadata)

            written = pd.read_csv(history_path)
            self.assertEqual(written.loc[0, "run_id"], "run-1")
            self.assertEqual(written.loc[0, "watchlist_label"], "test-watchlist")
            self.assertEqual(written.loc[0, "symbol"], "AAPL")

    def test_append_user_favorites_logs_against_current_bot_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "user_favorites.csv"
            result = append_master_avwap_user_favorites(
                "AAPL\nTSLA",
                source="test",
                output_context={
                    "run_date": "2026-04-24",
                    "scan_generated_at": "2026-04-24T12:00:00",
                    "favorites": "AAPL, MSFT",
                    "near_favorites": "NVDA",
                    "long_focus": "AAPL, MSFT, NVDA",
                    "short_focus": "",
                },
                path=feedback_path,
            )

            self.assertTrue(result["saved"])
            written = pd.read_csv(feedback_path)
            self.assertEqual(written.loc[0, "user_symbols"], "AAPL, TSLA")
            self.assertEqual(written.loc[0, "overlap_symbols"], "AAPL")
            self.assertEqual(written.loc[0, "missing_from_bot_symbols"], "TSLA")
            self.assertIn("MSFT", written.loc[0, "bot_not_selected_symbols"])


if __name__ == "__main__":
    unittest.main()
