import json
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
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
    analyze_avwap_retest_behavior,
    analyze_mid_earnings_ema_retest_setup,
    analyze_sma_breakout_setup,
    analyze_top_pattern_setup,
    append_d1_feature_history,
    append_master_avwap_user_favorites,
    apply_final_priority_buckets,
    apply_priority_rejection_score_caps,
    apply_tracker_setup_type_adjustments,
    apply_tracker_scoring_guardrails,
    apply_recent_tracker_setup_family_adjustments,
    assess_daily_relative_strength,
    attach_setup_candidate_payloads,
    build_market_prep_payload,
    build_combined_avwap_output_text,
    build_priority_setup_summary,
    build_master_avwap_focus_setup_type_text,
    build_master_avwap_focus_side_groups,
    build_recent_tracker_setup_family_rows,
    build_bot_tier_catch_rate_rows,
    build_bot_tier_outcome_rows,
    build_bot_tier_performance_rows,
    build_bot_tier_pick_rows,
    build_scan_factor_leaderboard_rows,
    build_scan_factor_observation_rows,
    compute_indicator_frame,
    evaluate_theta_pcs_candidate,
    evaluate_theta_put_candidate,
    export_bot_tier_tracker_views,
    export_scan_factor_views,
    extract_theta_rows_from_report,
    extract_theta_reason_risk_rows,
    extract_theta_symbols_from_report,
    format_market_prep_payload_report,
    load_scan_earnings_context,
    rank_tracker_setup_type_rows,
    update_master_avwap_d1_watchlist,
    write_stdev_range_report,
    write_theta_put_report,
    write_priority_setup_report,
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


def _build_daily_bar_cache_frame(days: int = 120) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp(datetime.now().date()), periods=days)
    rows = []
    for idx, dt_value in enumerate(dates):
        close = 100.0 + idx * 0.1
        rows.append(
            {
                "datetime": dt_value,
                "open": close - 0.2,
                "high": close + 0.8,
                "low": close - 0.8,
                "close": close,
                "volume": 1_000_000 + idx,
            }
        )
    return pd.DataFrame(rows)


def _build_intraday_htf_retest_frame(hours: int = 360) -> pd.DataFrame:
    datetimes = pd.date_range("2026-04-01 09:30", periods=hours, freq="h")
    rows = []
    for idx, dt_value in enumerate(datetimes):
        close = 100.0 + idx * 0.08
        rows.append(
            {
                "datetime": dt_value,
                "open": close - 0.05,
                "high": close + 0.35,
                "low": close - 0.35,
                "close": close,
                "volume": 250_000 + idx,
            }
        )
    frame = pd.DataFrame(rows)
    four_hour = master_avwap.resample_intraday_bars_to_4h(frame)
    four_hour_indicators = master_avwap.compute_indicator_frame(four_hour)
    four_hour_sma20 = float(four_hour_indicators.iloc[-1]["sma_20"])
    frame.loc[frame.index[-4:], "low"] = four_hour_sma20 - 0.05
    return frame


def _build_post_earnings_52w_history(
    *,
    side: str = "LONG",
    extra_sessions_after_break: int = 0,
) -> pd.DataFrame:
    gap_idx = 30
    periods = gap_idx + 2 + extra_sessions_after_break
    dates = pd.bdate_range("2026-01-02", periods=periods)
    rows = []
    for idx, dt_value in enumerate(dates):
        if side == "LONG":
            close_price = 112.0 + (idx * 0.24)
            open_price = close_price - 0.15
            high_price = close_price + 0.35
            low_price = close_price - 1.0
            if idx == gap_idx:
                open_price = 119.25
                high_price = 120.0
                low_price = 117.5
                close_price = 119.7
            elif idx == gap_idx + 1:
                open_price = 119.0
                high_price = 121.0
                low_price = 118.2
                close_price = 120.4
            elif idx > gap_idx + 1:
                age = idx - (gap_idx + 1)
                open_price = 118.5 - (age * 0.4)
                high_price = 119.0 - (age * 0.2)
                low_price = 116.4 - (age * 0.6)
                close_price = 117.6 - (age * 0.7)
        else:
            close_price = 88.0 - (idx * 0.24)
            open_price = close_price + 0.15
            high_price = close_price + 1.0
            low_price = close_price - 0.35
            if idx == gap_idx:
                open_price = 80.75
                high_price = 82.0
                low_price = 80.0
                close_price = 80.3
            elif idx == gap_idx + 1:
                open_price = 80.8
                high_price = 81.2
                low_price = 79.0
                close_price = 79.6
            elif idx > gap_idx + 1:
                age = idx - (gap_idx + 1)
                open_price = 81.0 + (age * 0.35)
                high_price = 82.1 + (age * 0.45)
                low_price = 80.5 + (age * 0.25)
                close_price = 81.2 + (age * 0.5)
        rows.append(
            {
                "datetime": dt_value,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": 1_500_000,
            }
        )
    return pd.DataFrame(rows)


def _build_post_earnings_52w_release_context(df: pd.DataFrame, *, side: str = "LONG") -> dict:
    gap_idx = 30
    anchor_idx = gap_idx - 1
    anchor_date = df.iloc[anchor_idx]["datetime"].date().isoformat()
    gap_date = df.iloc[gap_idx]["datetime"].date().isoformat()
    return {
        "active": True,
        "earnings_date": anchor_date,
        "release_session": "amc",
        "gap_date": gap_date,
        "anchor_date": anchor_date,
        "gap_idx": gap_idx,
        "anchor_idx": anchor_idx,
        "gap_atr_multiple": 1.25,
        "gap_is_up": side == "LONG",
        "gap_is_down": side != "LONG",
        "sessions_since_gap": len(df) - gap_idx - 1,
        "in_post_earnings_window": True,
        "anchor_meta": {
            "date": anchor_date,
            "vwap": 115.0 if side == "LONG" else 85.0,
            "stdev": 1.0,
            "bands": {},
        },
    }


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
        open_price = close_price - 1.0 if side == "LONG" else close_price + 1.0
        rows.append(
            {
                "datetime": dt_value,
                "open": open_price,
                "high": close_price + 1.0,
                "low": close_price - 2.0,
                "close": close_price,
                "volume": 1_500_000,
            }
        )
    return pd.DataFrame(rows)


def _build_sma_breakout_history(*, confirmed: bool = True) -> pd.DataFrame:
    dates = pd.bdate_range("2025-07-01", periods=220)
    rows = []
    for idx, dt_value in enumerate(dates):
        close_price = 100.0
        if idx == 213:
            close_price = 99.0
        elif idx == 214:
            close_price = 104.0
        elif idx == 215:
            close_price = 102.8
        elif idx in {216, 217, 218}:
            close_price = 103.5 + ((idx - 216) * 0.4)
        elif idx == 219:
            close_price = 106.0 if confirmed else 99.5

        rows.append(
            {
                "datetime": dt_value,
                "open": close_price - 0.4,
                "high": close_price + 5.0,
                "low": close_price - 5.0,
                "close": close_price,
                "volume": 1_500_000,
            }
        )

    df = pd.DataFrame(rows)
    indicator_frame = master_avwap.compute_indicator_frame(df)
    retest_idx = 215
    retest_level = float(indicator_frame.iloc[retest_idx]["ema_15"])
    df.loc[retest_idx, "low"] = retest_level + 0.02
    return df


def _build_top_pattern_history() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=360)
    rows = []
    for idx, dt_value in enumerate(dates):
        close_price = 20.0 + (idx * 0.40)
        rows.append(
            {
                "datetime": dt_value,
                "open": close_price - 0.6,
                "high": close_price + 1.2,
                "low": close_price - 1.2,
                "close": close_price,
                "volume": 1_500_000,
            }
        )

    df = pd.DataFrame(rows)
    indicator_frame = master_avwap.compute_indicator_frame(df)
    prev_sma50 = float(indicator_frame.iloc[-2]["sma_50"])
    latest_sma50 = float(indicator_frame.iloc[-1]["sma_50"])

    df.loc[df.index[-2], "open"] = prev_sma50 + 0.5
    df.loc[df.index[-2], "high"] = prev_sma50 + 1.0
    df.loc[df.index[-2], "low"] = prev_sma50 - 3.0
    df.loc[df.index[-2], "close"] = prev_sma50 - 2.0

    df.loc[df.index[-1], "open"] = latest_sma50 - 0.8
    df.loc[df.index[-1], "high"] = latest_sma50 + 2.2
    df.loc[df.index[-1], "low"] = latest_sma50 - 0.4
    df.loc[df.index[-1], "close"] = latest_sma50 + 1.0
    return df


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
        row["datetime"].date().isoformat(): {"vwap": 114.0, "bands": dict(bands)}
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
    favorite_zone: str = "",
    setup_tags: list[str] | None = None,
    favorite_signals: list[str] | None = None,
    retest_followthrough: bool = False,
    retest_reference_level: str = "",
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
        "favorite_zone": favorite_zone,
        "setup_tags": list(setup_tags or []),
        "favorite_signals": list(favorite_signals or []),
        "retest_followthrough": bool(retest_followthrough),
        "retest_reference_level": retest_reference_level,
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


def _build_avwape_retest_daily_rows() -> list[dict]:
    dates = pd.bdate_range("2026-04-01", periods=6)
    rows = []
    for idx, dt_value in enumerate(dates):
        close_price = 103.0 + (idx * 0.4)
        rows.append(
            {
                "date": dt_value.date().isoformat(),
                "open": close_price - 0.2,
                "high": close_price + 0.8,
                "low": close_price - 0.8,
                "close": close_price,
                "volume": 1_000_000,
            }
        )
    return rows


class MasterAvwapSetupTests(unittest.TestCase):
    def test_avwap_retest_followthrough_rejects_overextended_long(self):
        rows = [
            {"date": "2026-05-18", "open": 99.0, "high": 102.0, "low": 98.0, "close": 100.0},
            {"date": "2026-05-19", "open": 100.0, "high": 103.0, "low": 99.2, "close": 101.0},
            {"date": "2026-05-20", "open": 101.0, "high": 103.0, "low": 99.8, "close": 102.0},
            {"date": "2026-05-21", "open": 102.0, "high": 105.0, "low": 101.0, "close": 103.0},
            {"date": "2026-05-22", "open": 104.0, "high": 116.0, "low": 103.5, "close": 115.0},
        ]

        result = analyze_avwap_retest_behavior(
            rows,
            date(2026, 5, 22),
            current_vwap=100.0,
            side="LONG",
            current_upper_1=108.0,
            current_lower_1=95.0,
            atr20=4.0,
        )

        self.assertFalse(result["retest_followthrough"])
        self.assertIn("overextended", result["retest_note"])

    def test_avwap_retest_followthrough_allows_actionable_long(self):
        rows = [
            {"date": "2026-05-18", "open": 99.0, "high": 102.0, "low": 98.0, "close": 100.0},
            {"date": "2026-05-19", "open": 100.0, "high": 103.0, "low": 99.2, "close": 101.0},
            {"date": "2026-05-20", "open": 101.0, "high": 103.0, "low": 99.8, "close": 102.0},
            {"date": "2026-05-21", "open": 102.0, "high": 105.0, "low": 101.0, "close": 103.0},
            {"date": "2026-05-22", "open": 104.0, "high": 108.6, "low": 103.5, "close": 107.8},
        ]

        result = analyze_avwap_retest_behavior(
            rows,
            date(2026, 5, 22),
            current_vwap=100.0,
            side="LONG",
            current_upper_1=108.0,
            current_lower_1=95.0,
            atr20=4.0,
        )

        self.assertTrue(result["retest_followthrough"])
        self.assertEqual(result["retest_reference_level"], "AVWAPE")

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

    def test_priority_breakouts_only_promote_avwape_and_first_dev(self):
        second_dev_summary = build_priority_setup_summary(
            symbol="AAPL",
            side="LONG",
            events_today=["CROSS_UP_UPPER_2"],
            all_events=["CROSS_UP_UPPER_2"],
            trend_label="UP",
            favorite_zone=None,
        )
        first_dev_summary = build_priority_setup_summary(
            symbol="NKE",
            side="SHORT",
            events_today=["CROSS_DOWN_LOWER_1"],
            all_events=["CROSS_DOWN_LOWER_1"],
            trend_label="DOWN",
            favorite_zone=None,
        )

        self.assertFalse(second_dev_summary["has_favorite_signal"])
        self.assertEqual(second_dev_summary["favorite_signals"], [])
        self.assertTrue(first_dev_summary["has_favorite_signal"])
        self.assertEqual(first_dev_summary["favorite_signals"], ["CROSS_DOWN_LOWER_1"])

    def test_priority_band_bounce_is_side_specific_avwape_or_first_dev(self):
        long_summary = build_priority_setup_summary(
            symbol="NVDA",
            side="LONG",
            events_today=["BOUNCE_UPPER_1", "BOUNCE_LOWER_1"],
            all_events=["BOUNCE_UPPER_1", "BOUNCE_LOWER_1"],
            trend_label="UP",
            favorite_zone=None,
        )
        short_summary = build_priority_setup_summary(
            symbol="TSLA",
            side="SHORT",
            events_today=["BOUNCE_LOWER_1", "BOUNCE_UPPER_1"],
            all_events=["BOUNCE_LOWER_1", "BOUNCE_UPPER_1"],
            trend_label="DOWN",
            favorite_zone=None,
        )

        self.assertEqual(long_summary["favorite_signals"], ["BOUNCE_UPPER_1"])
        self.assertEqual(long_summary["setup_family"], "avwap_band_bounce")
        self.assertEqual(short_summary["favorite_signals"], ["BOUNCE_LOWER_1"])
        self.assertEqual(short_summary["setup_family"], "avwap_band_bounce")

    def test_avwape_to_first_dev_family_starts_after_ten_post_earnings_sessions(self):
        early_summary = build_priority_setup_summary(
            symbol="NOK",
            side="LONG",
            events_today=["CROSS_UP_UPPER_1"],
            all_events=["CROSS_UP_UPPER_1"],
            trend_label="UP",
            favorite_zone="AVWAPE to UPPER_1",
            post_earnings_sessions_since_gap=9,
        )
        ready_summary = build_priority_setup_summary(
            symbol="NOK",
            side="LONG",
            events_today=["CROSS_UP_UPPER_1"],
            all_events=["CROSS_UP_UPPER_1"],
            trend_label="UP",
            favorite_zone="AVWAPE to UPPER_1",
            post_earnings_sessions_since_gap=10,
        )
        retest_summary = build_priority_setup_summary(
            symbol="NOK",
            side="LONG",
            events_today=[],
            all_events=[],
            trend_label="UP",
            favorite_zone="AVWAPE to UPPER_1",
            retest_followthrough=True,
            retest_reference_level="AVWAPE",
            post_earnings_sessions_since_gap=10,
        )

        self.assertEqual(early_summary["setup_family"], "avwap_breakout")
        self.assertEqual(ready_summary["setup_family"], master_avwap.AVWAPE_TO_FIRST_DEV_FAMILY)
        self.assertEqual(retest_summary["setup_family"], master_avwap.AVWAPE_TO_FIRST_DEV_FAMILY)

    def test_avwape_retest_detects_same_day_long_bounce(self):
        rows = _build_avwape_retest_daily_rows()
        rows[-1].update({"open": 99.8, "high": 101.2, "low": 99.2, "close": 100.7})

        result = master_avwap.analyze_avwap_retest_behavior(
            rows,
            date.fromisoformat(rows[-1]["date"]),
            current_vwap=100.0,
            side="LONG",
            current_upper_1=104.0,
            current_lower_1=96.0,
            atr20=4.0,
        )

        self.assertTrue(result["retest_followthrough"])
        self.assertEqual(result["retest_reference_level"], "AVWAPE")
        self.assertIn("today", result["retest_note"])

    def test_avwape_retest_detects_prior_day_short_rejection(self):
        rows = _build_avwape_retest_daily_rows()
        rows[-2].update({"open": 100.4, "high": 100.8, "low": 98.8, "close": 99.4})
        rows[-1].update({"open": 98.8, "high": 98.9, "low": 98.1, "close": 98.6})

        result = master_avwap.analyze_avwap_retest_behavior(
            rows,
            date.fromisoformat(rows[-1]["date"]),
            current_vwap=100.0,
            side="SHORT",
            current_upper_1=104.0,
            current_lower_1=96.0,
            atr20=4.0,
        )

        self.assertTrue(result["retest_followthrough"])
        self.assertEqual(result["retest_reference_level"], "AVWAPE")
        self.assertIn("last session", result["retest_note"])

    def test_avwape_retest_ignores_bounces_older_than_two_sessions(self):
        rows = _build_avwape_retest_daily_rows()
        rows[-3].update({"open": 99.8, "high": 101.2, "low": 99.2, "close": 100.7})
        rows[-2].update({"open": 102.0, "high": 103.2, "low": 101.7, "close": 102.8})
        rows[-1].update({"open": 103.0, "high": 104.4, "low": 102.6, "close": 104.0})

        result = master_avwap.analyze_avwap_retest_behavior(
            rows,
            date.fromisoformat(rows[-1]["date"]),
            current_vwap=100.0,
            side="LONG",
            current_upper_1=104.0,
            current_lower_1=96.0,
            atr20=4.0,
        )

        self.assertFalse(result["retest_followthrough"])
        self.assertEqual(result["retest_reference_level"], "")

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
                intraday_vwap=114.0,
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

    def test_mid_earnings_retester_can_start_at_ten_sessions_after_gap(self):
        df = _build_mid_earnings_retest_history("LONG")
        release_context = _build_mid_earnings_release_context(df, sessions_since_gap=10)
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
        self.assertEqual(result["sessions_since_gap"], 10)
        self.assertEqual(result["primary_trigger_level"], "EMA_21")

    def test_sma_breakout_setup_flags_200sma_reclaim_after_ema15_retest(self):
        df = _build_sma_breakout_history(confirmed=True)
        indicator_frame = compute_indicator_frame(df)

        result = analyze_sma_breakout_setup(
            df,
            "LONG",
            indicator_frame=indicator_frame,
            atr20=10.0,
        )

        self.assertTrue(result["watch"])
        self.assertTrue(result["confirmed"])
        self.assertEqual(result["setup_family"], master_avwap.SMA_BREAKOUT_FAMILY)
        self.assertEqual(result["signal"], "SMA_BREAKOUT_200_RECLAIM")
        self.assertEqual(result["breakout_sma_label"], "SMA_200")
        self.assertEqual(result["breakout_age_sessions"], 5)
        self.assertEqual(result["retest_level"], "EMA_15")
        self.assertTrue(result["higher_high_confirmed"])
        self.assertTrue(result["previous_day_high_break"])
        self.assertEqual(result["confirmation_trigger"], "previous_day_high_break")
        self.assertIn("latest close reclaimed SMA_200", result["note"])
        self.assertIn("broke prior-day high", result["note"])

    def test_sma_breakout_setup_waits_for_higher_high_after_sma_reclaim(self):
        df = _build_sma_breakout_history(confirmed=True)
        df.loc[219, "high"] = float(df.loc[218, "high"]) - 0.5
        indicator_frame = compute_indicator_frame(df)

        result = analyze_sma_breakout_setup(
            df,
            "LONG",
            indicator_frame=indicator_frame,
            atr20=10.0,
        )

        self.assertTrue(result["watch"])
        self.assertTrue(result["tracking"])
        self.assertFalse(result["confirmed"])
        self.assertFalse(result["higher_high_confirmed"])
        self.assertFalse(result["previous_day_high_break"])
        self.assertEqual(result["setup_family"], master_avwap.SMA_BREAKOUT_TRACKING_FAMILY)
        self.assertEqual(result["signal"], "")
        self.assertIn("awaiting higher high / prior-day high break", result["note"])

    def test_sma_breakout_setup_tracks_retest_until_sma_reclaim(self):
        df = _build_sma_breakout_history(confirmed=False)
        indicator_frame = compute_indicator_frame(df)

        result = analyze_sma_breakout_setup(
            df,
            "LONG",
            indicator_frame=indicator_frame,
            atr20=10.0,
        )

        self.assertTrue(result["watch"])
        self.assertTrue(result["tracking"])
        self.assertFalse(result["confirmed"])
        self.assertEqual(result["setup_family"], master_avwap.SMA_BREAKOUT_TRACKING_FAMILY)
        self.assertEqual(result["signal"], "")
        self.assertIn("awaiting close back above SMA_200", result["note"])

    def test_sma_breakout_setup_rejects_one_day_sma_undercut_reclaim(self):
        df = _build_sma_breakout_history(confirmed=True)
        for idx in range(205, 213):
            df.loc[idx, "close"] = 105.0
            df.loc[idx, "open"] = 104.6
            df.loc[idx, "high"] = 106.0
            df.loc[idx, "low"] = 104.0
        indicator_frame = compute_indicator_frame(df)

        result = analyze_sma_breakout_setup(
            df,
            "LONG",
            indicator_frame=indicator_frame,
            atr20=10.0,
        )

        self.assertFalse(result["watch"])
        self.assertFalse(result["confirmed"])

    def test_sma_breakout_setup_requires_full_200sma_history_window(self):
        df = _build_sma_breakout_history(confirmed=True).tail(master_avwap.SMA_BREAKOUT_MIN_HISTORY_BARS - 1)
        indicator_frame = compute_indicator_frame(df)

        result = analyze_sma_breakout_setup(
            df,
            "LONG",
            indicator_frame=indicator_frame,
            atr20=10.0,
        )

        self.assertFalse(result["watch"])

    def test_top_pattern_setup_detects_weekly_leader_daily_50sma_entry(self):
        df = _build_top_pattern_history()
        indicator_frame = compute_indicator_frame(df)

        result = analyze_top_pattern_setup(
            df,
            side="LONG",
            indicator_frame=indicator_frame,
            atr20=3.0,
        )

        self.assertTrue(result["watch"])
        self.assertTrue(result["favorite_signal"])
        self.assertEqual(result["setup_family"], master_avwap.TOP_PATTERN_FAMILY)
        self.assertEqual(result["signal"], master_avwap.TOP_PATTERN_SIGNAL)
        self.assertTrue(result["weekly_ema15_hold"])
        self.assertGreaterEqual(result["weekly_ema15_hold_ratio"], master_avwap.TOP_PATTERN_WEEKLY_EMA15_MIN_RATIO)
        self.assertGreater(result["weekly_return_26w_pct"], master_avwap.TOP_PATTERN_WEEKLY_MIN_26W_RETURN_PCT)
        self.assertIn("daily_50sma_reclaim", result["entry_triggers"])
        self.assertTrue(result["daily_sma50_bounce"])
        self.assertIn("entry=", result["note"])

    def test_top_pattern_entry_remains_favorite_tracker_family(self):
        row = {
            "symbol": "TOPS",
            "side": "LONG",
            "priority_bucket": "favorite_setup",
            "setup_family": master_avwap.TOP_PATTERN_FAMILY,
            "top_pattern_watch": True,
            "top_pattern_entry": True,
        }

        self.assertEqual(master_avwap._tracker_priority_bucket(row), "favorite_setup")

    def test_top_pattern_tracker_attributes_tag_recent_sma_bounce_timing(self):
        attributes, _ = master_avwap.build_tracker_entry_attributes(
            {
                "side": "LONG",
                "symbol": "TOPS",
                "priority_bucket": "favorite_setup",
                "setup_family": master_avwap.TOP_PATTERN_FAMILY,
                "top_pattern_watch": True,
                "top_pattern_entry": True,
                "top_pattern_daily_sma50_bounce": True,
                "top_pattern_weekly_sma50_entry": True,
                "top_pattern_avwape_retest": True,
            },
            {"side": "LONG", "symbol": "TOPS"},
            {},
        )

        self.assertTrue(attributes["setup.top_pattern_recent_sma_bounce_entry"])
        self.assertEqual(
            attributes["setup.top_pattern_entry_timing_styles"],
            ["daily_50sma_bounce", "weekly_50sma_test", "avwape_retest"],
        )
        self.assertTrue(attributes["pattern.top_pattern_recent_sma_bounce_entry"])
        self.assertTrue(attributes["pattern.top_pattern_weekly_sma50_entry"])

    def test_top_pattern_is_secondary_to_a_real_entry_pattern(self):
        # TOP entry + a real SMA breakout: family is the SMA breakout (so the tracker
        # studies that pattern), NOT collapsed into top_pattern (Feat 2).
        family, _ = master_avwap._derive_setup_family(
            [],
            side="LONG",
            retest_followthrough=False,
            retest_reference_level="",
            extreme_move_favorite_ready=False,
            mid_earnings_active_second_stdev_hold=False,
            mid_earnings_primary_trigger_level="",
            sma_breakout_confirmed=True,
            top_pattern_entry=True,
            favorite_zone=None,
        )
        self.assertEqual(family, master_avwap.SMA_BREAKOUT_FAMILY)

    def test_top_pattern_is_primary_only_when_no_other_pattern_fired(self):
        family, _ = master_avwap._derive_setup_family(
            [],
            side="LONG",
            retest_followthrough=False,
            retest_reference_level="",
            extreme_move_favorite_ready=False,
            mid_earnings_active_second_stdev_hold=False,
            mid_earnings_primary_trigger_level="",
            top_pattern_entry=True,
            favorite_zone=None,
        )
        self.assertEqual(family, master_avwap.TOP_PATTERN_FAMILY)

    def test_top_pattern_watch_demotes_below_sma_breakout_tracking(self):
        family, _ = master_avwap._derive_setup_family(
            [],
            side="LONG",
            retest_followthrough=False,
            retest_reference_level="",
            extreme_move_favorite_ready=False,
            mid_earnings_active_second_stdev_hold=False,
            mid_earnings_primary_trigger_level="",
            sma_breakout_watch=True,
            top_pattern_watch=True,
            favorite_zone=None,
        )
        self.assertEqual(family, master_avwap.SMA_BREAKOUT_TRACKING_FAMILY)

    def test_top_secondary_flag_and_bonus_apply_when_demoted(self):
        base = master_avwap.build_priority_setup_summary(
            "TOPS", "LONG", [], [], "UPTREND", None,
            sma_breakout_confirmed=True,
        )
        boosted = master_avwap.build_priority_setup_summary(
            "TOPS", "LONG", [], [], "UPTREND", None,
            sma_breakout_confirmed=True,
            top_pattern_entry=True,
            top_pattern_score_bonus=38,
        )
        # Real pattern stays the primary family; TOP rides along as secondary + bonus.
        self.assertEqual(boosted["setup_family"], master_avwap.SMA_BREAKOUT_FAMILY)
        self.assertTrue(boosted["top_secondary"])
        self.assertFalse(base["top_secondary"])
        self.assertEqual(boosted["top_pattern_score_bonus"], 38)
        self.assertGreater(boosted["score"], base["score"])

    def test_top_only_setup_keeps_top_family_and_not_secondary(self):
        top_only = master_avwap.build_priority_setup_summary(
            "TOPS", "LONG", [], [], "UPTREND", None,
            top_pattern_entry=True,
            top_pattern_score_bonus=38,
        )
        self.assertEqual(top_only["setup_family"], master_avwap.TOP_PATTERN_FAMILY)
        self.assertFalse(top_only["top_secondary"])

    def test_daily_relative_strength_prefers_long_strength_and_short_weakness_vs_spy(self):
        dates = pd.bdate_range("2026-06-01", periods=6)
        strong_rows = [
            {"date": dt_value.date().isoformat(), "close": close}
            for dt_value, close in zip(dates, [100.0, 101.0, 102.0, 103.0, 104.0, 108.0])
        ]
        weak_rows = [
            {"date": dt_value.date().isoformat(), "close": close}
            for dt_value, close in zip(dates, [100.0, 99.0, 98.0, 97.0, 96.0, 92.0])
        ]
        spy = {"one_day_return_pct": 0.5, "five_day_return_pct": 1.0}

        long_result = assess_daily_relative_strength(strong_rows, dates[-1].date(), "LONG", spy)
        short_result = assess_daily_relative_strength(weak_rows, dates[-1].date(), "SHORT", spy)

        self.assertEqual(long_result["daily_relative_strength_bonus"], master_avwap.DAILY_RELATIVE_STRENGTH_BONUS)
        self.assertIn("stronger than SPY", long_result["daily_relative_strength_note"])
        self.assertEqual(short_result["daily_relative_strength_bonus"], master_avwap.DAILY_RELATIVE_STRENGTH_WEAKNESS_BONUS)
        self.assertIn("weaker than SPY", short_result["daily_relative_strength_note"])

    def test_ibkr_daily_fetch_rounds_year_duration_up(self):
        class FakeIB:
            def __init__(self):
                self.data = {}
                self.ready = {}
                self.historical_symbols = {}
                self.duration = None

            def reqHistoricalData(self, req_id, _contract, _end, duration, *_args):
                self.duration = duration
                self.data[req_id] = [
                    {
                        "time": "20260102",
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.0,
                        "close": 10.5,
                        "volume": 1000,
                    }
                ]
                self.ready[req_id] = True

        fake_ib = FakeIB()
        result = master_avwap._fetch_live_daily_bars(fake_ib, "AAPL", 420)

        self.assertEqual(fake_ib.duration, "2 Y")
        self.assertFalse(result.empty)
        self.assertEqual(result.attrs.get(master_avwap.DAILY_BAR_SOURCE_ATTR), master_avwap.DAILY_BAR_SOURCE_IBKR)
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

    def test_mid_earnings_retest_requires_strength_confirmation(self):
        df = _build_mid_earnings_retest_history("LONG")
        df.loc[df.index[-1], "open"] = df.loc[df.index[-1], "close"] + 1.0
        release_context = _build_mid_earnings_release_context(df, sessions_since_gap=16)
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
                intraday_vwap=114.0,
            )

        self.assertTrue(result["watch"])
        self.assertFalse(result["favorite_signal"])
        self.assertEqual(result["events"], [])
        self.assertIn("waiting for close above open and intraday VWAP", result["note"])

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
                intraday_vwap=114.0,
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
        self.assertIn("Current 2nd/3rd stdev retest tracking list", text)
        self.assertIn("Longs tracked for later retest: AAPL", text)
        self.assertIn("Shorts tracked for later retest: TSLA", text)
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

    def test_focus_setup_type_text_ranks_symbols_by_total_score(self):
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
                    "AAPL LONG score=99 family=avwap_breakout bucket=favorite_setup",
                    "TSLA SHORT score=97 family=avwap_breakdown bucket=favorite_setup",
                    "MSFT LONG score=95 family=avwap_breakout bucket=near_favorite_zone",
                    "NVDA LONG score=94 family=post_earnings_52w_break bucket=post_earnings_play",
                ]
            ),
        )

    def test_priority_setup_report_groups_copy_lists_by_setup_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "priority_report.txt"
            write_priority_setup_report(
                report_path,
                [
                    {
                        "symbol": "AAPL",
                        "side": "LONG",
                        "score": 101,
                        "priority_bucket": "favorite_setup",
                        "setup_family": "avwap_breakout",
                        "favorite_signals": ["CROSS_UP_VWAP"],
                        "context_signals": [],
                        "favorite_zone": "AVWAPE to UPPER_1",
                        "current_band_zone": "VWAP to UPPER_1",
                        "trend_20d": "UP",
                    },
                    {
                        "symbol": "TSLA",
                        "side": "SHORT",
                        "score": 98,
                        "priority_bucket": "near_favorite_zone",
                        "setup_family": "avwap_breakdown",
                        "favorite_signals": [],
                        "context_signals": ["BOUNCE_VWAP"],
                        "favorite_zone": "LOWER_1 to AVWAPE",
                        "current_band_zone": "VWAP to LOWER_1",
                        "trend_20d": "DOWN",
                    },
                ],
            )

            text = report_path.read_text(encoding="utf-8")

        self.assertIn("Copy/paste lists", text)
        self.assertIn("High conviction shortlist", text)
        self.assertIn("Best swing trades today", text)
        self.assertIn("LONG: AAPL", text)
        self.assertIn("SHORT: TSLA", text)
        self.assertIn("By setup type", text)
        self.assertLess(
            text.index("AAPL LONG ExpR=n/a score=101"),
            text.index("TSLA SHORT ExpR=n/a score=98"),
        )
        self.assertIn("AVWAP breakout\n  LONG: AAPL", text)
        self.assertIn("AVWAP breakdown\n  LONG: None\n  SHORT: TSLA", text)
        self.assertIn("Overall score rankings", text)
        self.assertLess(text.index("Overall score rankings"), text.index("Detailed setup notes"))
        ranking_block = text[text.index("Overall score rankings"):text.index("Detailed setup notes")]
        self.assertIn("1. AAPL", ranking_block)
        self.assertIn("2. TSLA", ranking_block)
        self.assertIn("Detailed setup notes", text)

    def test_priority_setup_report_best_trades_excludes_gated_short(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "priority_report.txt"
            write_priority_setup_report(
                report_path,
                [
                    {
                        "symbol": "AAPL",
                        "side": "LONG",
                        "score": 120,
                        "priority_bucket": "favorite_setup",
                        "setup_family": "avwap_breakout",
                        "favorite_signals": ["CROSS_UP_VWAP"],
                        "context_signals": [],
                        "favorite_zone": "AVWAPE to UPPER_1",
                        "current_band_zone": "VWAP to UPPER_1",
                        "previous_day_range_break": True,
                        "trend_20d": "UP",
                    },
                    {
                        "symbol": "TSLA",
                        "side": "SHORT",
                        "score": 118,
                        "priority_bucket": "near_favorite_zone",
                        "setup_family": "avwap_breakdown",
                        "favorite_signals": ["CROSS_DOWN_VWAP"],
                        "context_signals": [],
                        "favorite_zone": "LOWER_1 to AVWAPE",
                        "current_band_zone": "VWAP to LOWER_1",
                        "trend_20d": "SIDEWAYS",
                        "short_near_favorite_gate_note": "Short near-favorite gate: needs downtrend alignment",
                    },
                ],
            )

            text = report_path.read_text(encoding="utf-8")

        detail_start = text.index("Best swing trades today", text.index("2nd/3rd stdev retest tracking"))
        best_detail_block = text[detail_start:text.index("Overall score rankings")]
        self.assertIn("AAPL", best_detail_block)
        self.assertNotIn("TSLA", best_detail_block)

    def test_priority_setup_report_outputs_stdev_tracking_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "priority_report.txt"
            write_priority_setup_report(
                report_path,
                [
                    {
                        "symbol": "EXT",
                        "side": "LONG",
                        "score": 88,
                        "priority_bucket": "",
                        "setup_family": "mid_earnings_above_2nd_stdev",
                        "favorite_signals": [],
                        "context_signals": [],
                        "current_band_zone": "UPPER_2 to UPPER_3",
                        "trend_20d": "UP",
                    }
                ],
            )

            text = report_path.read_text(encoding="utf-8")

        self.assertIn("2nd/3rd stdev retest tracking", text)
        self.assertIn("LONG: EXT", text)
        ranking_block = text[text.index("Overall score rankings"):text.index("Detailed setup notes")]
        self.assertNotIn("EXT", ranking_block)
        self.assertNotIn("bucket=stdev-track", ranking_block)

    def test_focus_feed_outputs_extended_stdev_tracker_for_bot(self):
        row = {
            "symbol": "EXT",
            "side": "LONG",
            "score": 88,
            "priority_bucket": "",
            "setup_family": "mid_earnings_above_2nd_stdev",
            "favorite_signals": [],
            "context_signals": [],
            "current_band_zone": "UPPER_2 to UPPER_3",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            feed_path = Path(temp_dir) / "focus.json"
            master_avwap.write_master_avwap_focus_feed(feed_path, [row], {"symbols": {"EXT": {}}})
            payload = json.loads(feed_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["stdev_retest_tracking"][0]["symbol"], "EXT")
        self.assertEqual(payload["symbols"]["EXT"]["priority_bucket"], "stdev_retest_tracking")

    def test_priority_report_and_focus_feed_include_sma_breakout_tracking(self):
        row = {
            "symbol": "SMAT",
            "side": "LONG",
            "score": 72,
            "priority_bucket": "",
            "setup_family": master_avwap.SMA_BREAKOUT_TRACKING_FAMILY,
            "favorite_signals": [],
            "context_signals": [],
            "sma_breakout_watch": True,
            "sma_breakout_confirmed": False,
            "sma_breakout_sma_label": "SMA_200",
            "sma_breakout_sma_period": 200,
            "sma_breakout_retest_level": "EMA_15",
            "sma_breakout_note": "Broke up through SMA_200; awaiting close back above SMA_200",
            "trend_20d": "UP",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "priority_report.txt"
            focus_path = Path(temp_dir) / "focus.json"
            write_priority_setup_report(report_path, [row])
            master_avwap.write_master_avwap_focus_feed(focus_path, [row], {"symbols": {"SMAT": {}}})
            report_text = report_path.read_text(encoding="utf-8")
            payload = json.loads(focus_path.read_text(encoding="utf-8"))

        self.assertIn("SMA breakout retest tracking", report_text)
        self.assertIn("LONG: SMAT", report_text)
        ranking_block = report_text[report_text.index("Overall score rankings"):report_text.index("Detailed setup notes")]
        self.assertNotIn("SMAT", ranking_block)
        self.assertNotIn("bucket=sma-track", ranking_block)
        self.assertEqual(payload["sma_breakout_tracking"][0]["symbol"], "SMAT")
        self.assertEqual(payload["symbols"]["SMAT"]["priority_bucket"], "sma_breakout_tracking")

    def test_priority_report_and_focus_feed_include_top_pattern_tracking(self):
        row = {
            "symbol": "TOPS",
            "side": "LONG",
            "score": 91,
            "priority_bucket": "",
            "setup_family": master_avwap.TOP_PATTERN_TRACKING_FAMILY,
            "favorite_signals": [],
            "context_signals": [],
            "top_pattern_watch": True,
            "top_pattern_entry": False,
            "top_pattern_weekly_return_26w_pct": 82.5,
            "top_pattern_note": "TOP weekly trend; tracking for 50SMA/AVWAPE entry",
            "trend_20d": "UP",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "priority_report.txt"
            focus_path = Path(temp_dir) / "focus.json"
            top_path = Path(temp_dir) / "top_strength.txt"
            write_priority_setup_report(report_path, [row])
            master_avwap.write_top_strength_watchlist_report(top_path, [row])
            master_avwap.write_master_avwap_focus_feed(focus_path, [row], {"symbols": {"TOPS": {}}})
            report_text = report_path.read_text(encoding="utf-8")
            top_text = top_path.read_text(encoding="utf-8")
            payload = json.loads(focus_path.read_text(encoding="utf-8"))

        self.assertIn("TOP strength watchlist", report_text)
        self.assertIn("TOP pattern tracking", report_text)
        self.assertIn("LONG: TOPS", report_text)
        self.assertIn("H1 15EMA long entry", top_text)
        self.assertIn("TOPS", top_text)
        ranking_block = report_text[report_text.index("Overall score rankings"):report_text.index("Detailed setup notes")]
        self.assertNotIn("TOPS", ranking_block)
        self.assertEqual(payload["top_strength_watchlist"][0]["symbol"], "TOPS")
        self.assertEqual(payload["symbols"]["TOPS"]["priority_bucket"], "top_strength_watchlist")
        self.assertEqual(payload["top_pattern_tracking"][0]["symbol"], "TOPS")
        self.assertEqual(payload["top_pattern_tracking"][0]["top_pattern_weekly_return_26w_pct"], 82.5)

    def test_final_priority_buckets_drop_compressed_extended_retest(self):
        row = {
            "symbol": "AAON",
            "side": "LONG",
            "score": 53,
            "setup_family": "avwap_retest_followthrough",
            "favorite_signals": [],
            "context_signals": [],
            "favorite_zone": "",
            "has_favorite_signal": False,
            "retest_followthrough": True,
            "previous_anchor_path_clear": True,
            "current_band_zone": "UPPER_2 to UPPER_3",
            "compression_flag": True,
            "compression_penalty": 8,
            "recent_second_band_test_days": 2,
        }
        ai_state = {"symbols": {"AAON": {"current_band_zone": "UPPER_2 to UPPER_3", "compression_flag": True}}}

        apply_final_priority_buckets([row], ai_state, [], {})

        self.assertEqual(row["priority_bucket"], "")
        self.assertFalse(row["is_favorite_setup"])

    def test_final_priority_buckets_do_not_promote_second_dev_breakout(self):
        row = {
            "symbol": "APLE",
            "side": "LONG",
            "score": 13,
            "setup_family": "avwap_breakout",
            "favorite_signals": ["CROSS_UP_UPPER_2"],
            "context_signals": [],
            "favorite_zone": "",
            "has_favorite_signal": True,
            "current_band_zone": "UPPER_2 to UPPER_3",
            "compression_flag": False,
            "compression_penalty": 0,
        }
        ai_state = {"symbols": {"APLE": {"current_band_zone": "UPPER_2 to UPPER_3"}}}

        apply_final_priority_buckets([row], ai_state, [], {})

        self.assertEqual(row["priority_bucket"], "")
        self.assertFalse(row["is_favorite_setup"])

    def test_final_priority_buckets_track_extended_mid_earnings_not_recommend(self):
        row = {
            "symbol": "MID",
            "side": "LONG",
            "score": 80,
            "setup_family": "mid_earnings_above_2nd_stdev",
            "favorite_signals": [],
            "context_signals": [],
            "has_favorite_signal": False,
            "mid_earnings_watch": True,
            "mid_earnings_active_second_stdev_hold": True,
            "current_band_zone": "UPPER_2 to UPPER_3",
            "compression_flag": False,
            "compression_penalty": 0,
        }
        ai_state = {"symbols": {"MID": {"current_band_zone": "UPPER_2 to UPPER_3"}}}

        apply_final_priority_buckets([row], ai_state, [], {})

        self.assertEqual(row["priority_bucket"], "")
        self.assertFalse(row["is_near_favorite_zone"])

    def test_confirmed_mid_earnings_retest_can_flag_after_extended_tracking(self):
        row = {
            "symbol": "MIDB",
            "side": "LONG",
            "score": 128,
            "setup_family": "mid_earnings_ema15_retest",
            "favorite_signals": ["MID_EARNINGS_EMA15_RETEST"],
            "context_signals": [],
            "has_favorite_signal": True,
            "mid_earnings_watch": True,
            "mid_earnings_ema15_trigger": True,
            "previous_day_range_break": True,
            "current_band_zone": "UPPER_1 to UPPER_2",
            "trend_20d": "UP",
            "current_day_open": 100.0,
            "last_close": 104.0,
            "intraday_vwap": 101.0,
        }
        ai_state = {"symbols": {"MIDB": {"current_band_zone": "UPPER_1 to UPPER_2"}}}

        apply_final_priority_buckets([row], ai_state, [], {})

        self.assertEqual(row["priority_bucket"], "favorite_setup")
        self.assertTrue(row["is_favorite_setup"])

    def test_side_opposite_day_downgrades_and_caps_long_favorite(self):
        row = {
            "symbol": "RED",
            "side": "LONG",
            "score": 226,
            "setup_family": "mid_earnings_ema15_retest",
            "favorite_signals": ["MID_EARNINGS_EMA15_RETEST", "MID_EARNINGS_FIRST_DEV_RETEST"],
            "context_signals": ["MID_EARNINGS_EMA8_CONFLUENCE"],
            "favorite_zone": "AVWAPE to UPPER_1",
            "has_favorite_signal": True,
            "mid_earnings_watch": True,
            "current_band_zone": "VWAP to UPPER_1",
            "trend_20d": "UP",
            "side_aligned_day": False,
            "current_day_change_pct": -1.1,
            "side_alignment_note": "Long setup closed down on day (-1.10%)",
        }
        ai_state = {"symbols": {"RED": {"current_band_zone": "VWAP to UPPER_1"}}}
        feature_row = {}

        apply_final_priority_buckets([row], ai_state, [], {"RED": feature_row})
        apply_priority_rejection_score_caps([row], ai_state, {"RED": feature_row})

        self.assertEqual(row["priority_bucket"], "near_favorite_zone")
        self.assertFalse(row["is_favorite_setup"])
        self.assertEqual(row["score"], 120)
        self.assertIn("down on day", row["rejection_score_cap_note"])

    def test_first_dev_play_without_strength_stays_near_favorite(self):
        row = {
            "symbol": "FDEV",
            "side": "LONG",
            "score": 180,
            "setup_family": "mid_earnings_1stdev_retest",
            "favorite_signals": ["MID_EARNINGS_FIRST_DEV_RETEST"],
            "context_signals": [],
            "has_favorite_signal": True,
            "mid_earnings_watch": True,
            "mid_earnings_first_dev_trigger": True,
            "current_band_zone": "VWAP to UPPER_1",
            "current_day_open": 101.0,
            "last_close": 100.5,
            "intraday_vwap": 99.0,
        }
        ai_state = {"symbols": {"FDEV": {"current_band_zone": "VWAP to UPPER_1"}}}

        apply_final_priority_buckets([row], ai_state, [], {})

        self.assertEqual(row["priority_bucket"], "near_favorite_zone")
        self.assertFalse(row["is_favorite_setup"])

    def test_first_dev_play_with_strength_can_be_favorite(self):
        row = {
            "symbol": "FDEV",
            "side": "LONG",
            "score": 180,
            "setup_family": "mid_earnings_1stdev_retest",
            "favorite_signals": ["MID_EARNINGS_FIRST_DEV_RETEST"],
            "context_signals": [],
            "has_favorite_signal": True,
            "mid_earnings_watch": True,
            "mid_earnings_first_dev_trigger": True,
            "previous_day_range_break": True,
            "current_band_zone": "VWAP to UPPER_1",
            "current_day_open": 99.0,
            "last_close": 101.0,
            "intraday_vwap": 100.0,
        }
        ai_state = {"symbols": {"FDEV": {"current_band_zone": "VWAP to UPPER_1"}}}

        apply_final_priority_buckets([row], ai_state, [], {})

        self.assertEqual(row["priority_bucket"], "favorite_setup")
        self.assertTrue(row["is_favorite_setup"])

    def test_final_priority_buckets_marks_avwape_retest_as_preferred_swing_focus(self):
        row = {
            "symbol": "AVWP",
            "side": "LONG",
            "score": 112,
            "setup_family": "avwap_retest_followthrough",
            "favorite_signals": [],
            "context_signals": [],
            "has_favorite_signal": False,
            "retest_followthrough": True,
            "retest_reference_level": "AVWAPE",
            "current_band_zone": "VWAP to UPPER_1",
            "trend_20d": "UP",
            "current_day_open": 100.0,
            "last_close": 103.0,
            "intraday_vwap": 101.0,
        }
        ai_state = {"symbols": {"AVWP": {"current_band_zone": "VWAP to UPPER_1"}}}
        feature_row = {}

        apply_final_priority_buckets([row], ai_state, [], {"AVWP": feature_row})

        self.assertEqual(row["priority_bucket"], "favorite_setup")
        self.assertTrue(row["is_favorite_setup"])
        self.assertTrue(row["preferred_swing_focus"])
        self.assertTrue(ai_state["symbols"]["AVWP"]["preferred_swing_focus"])
        self.assertTrue(feature_row["preferred_swing_focus"])

    def test_final_priority_buckets_keeps_sma_reclaim_without_higher_high_unfocused(self):
        row = {
            "symbol": "SMAH",
            "side": "LONG",
            "score": 120,
            "setup_family": master_avwap.SMA_BREAKOUT_FAMILY,
            "favorite_signals": ["SMA_BREAKOUT_200_RECLAIM"],
            "context_signals": [],
            "has_favorite_signal": True,
            "sma_breakout_watch": True,
            "sma_breakout_confirmed": True,
            "sma_breakout_higher_high_confirmed": False,
            "sma_breakout_previous_day_high_break": False,
            "current_band_zone": "VWAP to UPPER_1",
            "trend_20d": "UP",
            "current_day_open": 100.0,
            "last_close": 103.0,
            "intraday_vwap": 101.0,
        }
        ai_state = {"symbols": {"SMAH": {"current_band_zone": "VWAP to UPPER_1"}}}

        apply_final_priority_buckets([row], ai_state, [], {})

        self.assertEqual(row["priority_bucket"], "")
        self.assertFalse(row["preferred_swing_focus"])

    def test_best_swing_trades_prefer_range_break_without_filtering_live_retest(self):
        rows = [
            {
                "symbol": "CECO",
                "side": "LONG",
                "score": 250,
                "priority_bucket": "favorite_setup",
                "setup_family": master_avwap.MID_EARNINGS_FIRST_DEV_RETEST_FAMILY,
                "favorite_signals": ["MID_EARNINGS_FIRST_DEV_RETEST"],
                "context_signals": [],
                "mid_earnings_first_dev_trigger": True,
                "previous_day_range_break": True,
                "current_band_zone": "UPPER_2 to UPPER_3",
                "current_day_open": 95.0,
                "last_close": 98.0,
                "intraday_vwap": 96.0,
            },
            {
                "symbol": "TECH",
                "side": "LONG",
                "score": 240,
                "priority_bucket": "favorite_setup",
                "setup_family": master_avwap.SMA_BREAKOUT_FAMILY,
                "favorite_signals": ["SMA_BREAKOUT_50_RECLAIM"],
                "context_signals": [],
                "sma_breakout_confirmed": True,
                "sma_breakout_previous_day_high_break": True,
                "sma_breakout_retest_level": "SMA_50",
                "current_band_zone": "UPPER_2 to UPPER_3",
            },
            {
                "symbol": "TGT",
                "side": "LONG",
                "score": 160,
                "priority_bucket": "favorite_setup",
                "setup_family": master_avwap.MID_EARNINGS_EMA15_RETEST_FAMILY,
                "favorite_signals": ["MID_EARNINGS_EMA15_RETEST", "MID_EARNINGS_FIRST_DEV_RETEST"],
                "context_signals": [],
                "mid_earnings_ema15_trigger": True,
                "mid_earnings_first_dev_trigger": True,
                "previous_day_range_break": False,
                "current_band_zone": "UPPER_1 to UPPER_2",
                "current_day_open": 129.0,
                "last_close": 130.7,
                "intraday_vwap": 130.5,
            },
            {
                "symbol": "VRT",
                "side": "LONG",
                "score": 211,
                "priority_bucket": "favorite_setup",
                "setup_family": "avwap_retest_followthrough",
                "favorite_signals": ["CROSS_UP_VWAP"],
                "context_signals": [],
                "retest_followthrough": True,
                "previous_day_range_break": True,
                "current_band_zone": "VWAP to UPPER_1",
            },
            {
                "symbol": "MOD",
                "side": "LONG",
                "score": 189,
                "priority_bucket": "favorite_setup",
                "setup_family": "avwap_retest_followthrough",
                "favorite_signals": ["CROSS_UP_VWAP"],
                "context_signals": [],
                "retest_followthrough": True,
                "previous_day_range_break": True,
                "current_band_zone": "VWAP to UPPER_1",
            },
            {
                "symbol": "TSHA",
                "side": "LONG",
                "score": 185,
                "priority_bucket": "near_favorite_zone",
                "setup_family": "avwap_retest_followthrough",
                "favorite_signals": ["CROSS_UP_UPPER_1"],
                "context_signals": ["PREV_BOUNCE_VWAP"],
                "retest_followthrough": True,
                "previous_day_range_break": True,
                "current_band_zone": "UPPER_1 to UPPER_2",
            },
            {
                "symbol": "LAUR",
                "side": "LONG",
                "score": 208,
                "priority_bucket": "favorite_setup",
                "setup_family": master_avwap.MID_EARNINGS_EMA15_RETEST_FAMILY,
                "favorite_signals": ["MID_EARNINGS_EMA15_RETEST", "MID_EARNINGS_FIRST_DEV_RETEST"],
                "context_signals": [],
                "mid_earnings_ema15_trigger": True,
                "mid_earnings_first_dev_trigger": True,
                "previous_day_range_break": True,
                "current_band_zone": "UPPER_1 to UPPER_2",
                "current_day_open": 36.1,
                "last_close": 36.43,
                "intraday_vwap": 36.12,
            },
        ]

        selected = master_avwap._priority_best_swing_trade_rows(rows)
        symbols = [row["symbol"] for row in selected]

        self.assertEqual(symbols, ["VRT", "LAUR", "MOD", "TSHA", "TGT"])
        self.assertNotIn("CECO", symbols)
        self.assertNotIn("TECH", symbols)

    def test_vwap_range_confirmation_requires_vwap_and_prior_day_trigger(self):
        long_summary = {
            "previous_day_high": 100.0,
            "previous_day_low": 95.0,
            "previous_day_range_break": True,
        }
        short_summary = {
            "previous_day_high": 105.0,
            "previous_day_low": 100.0,
            "previous_day_range_break": True,
        }

        long_result = master_avwap.assess_vwap_range_confirmation(101.0, 100.5, "LONG", long_summary)
        short_result = master_avwap.assess_vwap_range_confirmation(99.0, 99.5, "SHORT", short_summary)
        no_vwap_result = master_avwap.assess_vwap_range_confirmation(101.0, None, "LONG", long_summary)
        below_prior_high_result = master_avwap.assess_vwap_range_confirmation(99.5, 99.0, "LONG", long_summary)

        self.assertTrue(long_result["vwap_range_confirmation"])
        self.assertEqual(
            long_result["vwap_range_confirmation_bonus"],
            master_avwap.PRIORITY_VWAP_RANGE_CONFIRMATION_SCORE_BONUS,
        )
        self.assertIn("above intraday VWAP", long_result["vwap_range_confirmation_note"])
        self.assertTrue(short_result["vwap_range_confirmation"])
        self.assertIn("below intraday VWAP", short_result["vwap_range_confirmation_note"])
        self.assertFalse(no_vwap_result["vwap_range_confirmation"])
        self.assertFalse(below_prior_high_result["vwap_range_confirmation"])

    def test_priority_score_prefers_vwap_plus_prior_day_confirmation_without_filtering(self):
        base_summary = build_priority_setup_summary(
            symbol="BASE",
            side="LONG",
            events_today=["MID_EARNINGS_EMA15_RETEST"],
            all_events=["MID_EARNINGS_EMA15_RETEST"],
            trend_label="UP",
            favorite_zone=None,
            previous_day_range_break=True,
            previous_day_range_break_bonus=master_avwap.PRIORITY_PREVIOUS_DAY_RANGE_BREAK_SCORE_BONUS,
        )
        confirmed_summary = build_priority_setup_summary(
            symbol="CONF",
            side="LONG",
            events_today=["MID_EARNINGS_EMA15_RETEST"],
            all_events=["MID_EARNINGS_EMA15_RETEST"],
            trend_label="UP",
            favorite_zone=None,
            previous_day_range_break=True,
            previous_day_range_break_bonus=master_avwap.PRIORITY_PREVIOUS_DAY_RANGE_BREAK_SCORE_BONUS,
            vwap_range_confirmation=True,
            vwap_range_confirmation_bonus=master_avwap.PRIORITY_VWAP_RANGE_CONFIRMATION_SCORE_BONUS,
            vwap_range_confirmation_note="Holding above intraday VWAP and previous day high",
        )

        self.assertEqual(
            confirmed_summary["score"] - base_summary["score"],
            master_avwap.PRIORITY_VWAP_RANGE_CONFIRMATION_SCORE_BONUS,
        )
        self.assertFalse(base_summary["vwap_range_confirmation"])
        self.assertTrue(confirmed_summary["vwap_range_confirmation"])
        self.assertEqual(
            confirmed_summary["vwap_range_confirmation_bonus"],
            master_avwap.PRIORITY_VWAP_RANGE_CONFIRMATION_SCORE_BONUS,
        )

    def test_post_earnings_break_ignores_extended_stdev_inside_window(self):
        row = {
            "symbol": "PE",
            "side": "LONG",
            "score": 150,
            "setup_family": "post_earnings_52w_break",
            "favorite_signals": ["POST_EARNINGS_52W_BREAK"],
            "context_signals": [],
            "has_favorite_signal": True,
            "post_earnings_active": True,
            "post_earnings_sessions_since_gap": 2,
            "post_earnings_break_intraday": True,
            "post_earnings_break_close": True,
            "post_earnings_bands_expanding": False,
            "compression_flag": True,
            "compression_penalty": 12,
            "current_band_zone": "UPPER_2 to UPPER_3",
        }
        ai_state = {
            "symbols": {
                "PE": {
                    "current_band_zone": "UPPER_2 to UPPER_3",
                    "compression_flag": True,
                }
            }
        }

        apply_final_priority_buckets([row], ai_state, [], {})
        self.assertEqual(row["priority_bucket"], "favorite_setup")

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
                    "post_earnings_sessions_since_gap": 3,
                    "post_earnings_break_intraday": True,
                    "post_earnings_break_close": True,
                    "post_earnings_bands_expanding": True,
                    "post_earnings_note": "new 52-week closing low; close confirmed",
                },
                {
                    "symbol": "ADBE",
                    "side": "SHORT",
                    "score": 118,
                    "setup_family": "post_earnings_avwap_bounce",
                    "favorite_signals": [master_avwap.POST_EARNINGS_BOUNCE_SIGNAL],
                    "context_signals": [],
                    "post_earnings_active": True,
                    "post_earnings_sessions_since_gap": 5,
                    "post_earnings_note": "pre-earnings AVWAPE rejection",
                },
                {
                    "symbol": "OLD",
                    "side": "LONG",
                    "score": 300,
                    "post_earnings_active": True,
                    "post_earnings_sessions_since_gap": 16,
                    "post_earnings_break_intraday": True,
                    "post_earnings_note": "stale post-earnings setup",
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
        self.assertEqual(sections["post_earnings_potential_plays"]["copy_text"], "ADBE, AMD")

        report = format_market_prep_payload_report(payload)

        self.assertIn("Longs: AVWAPE to 1st Dev", report)
        self.assertIn("NFLX   last_night earnings=2026-04-23", report)
        self.assertIn("AMD    SHORT close_confirmed score=121.5", report)
        self.assertIn("ADBE   SHORT watch score=118.0 | pre-earnings AVWAPE rejection", report)

    def test_market_prep_payload_builds_strength_and_weakness_deciles(self):
        symbols = [
            "AAA",
            "BBB",
            "CCC",
            "DDD",
            "EEE",
            "FFF",
            "GGG",
            "HHH",
            "III",
            "JJJ",
            "KKK",
            "LLL",
            "MMM",
            "NNN",
            "OOO",
            "PPP",
            "QQQ",
            "RRR",
            "SSS",
            "TTT",
        ]
        priority_rows = []
        for idx, symbol in enumerate(symbols):
            rs_score = 10.0 - idx
            priority_rows.append(
                {
                    "symbol": symbol,
                    "side": "LONG" if rs_score >= 0 else "SHORT",
                    "score": 100 - idx,
                    "setup_family": "avwap_retest_followthrough",
                    "daily_relative_strength_score": rs_score,
                    "symbol_one_day_return_pct": rs_score / 2,
                    "symbol_five_day_return_pct": rs_score,
                }
            )

        payload = build_market_prep_payload(
            priority_rows=priority_rows,
            reference_date=date(2026, 4, 24),
            previous_session_date=date(2026, 4, 23),
            calendar_rows_by_date={},
        )
        sections = {section["id"]: section for section in payload["sections"]}

        strongest = sections["strongest_stocks_top_decile"]
        weakest = sections["weakest_stocks_bottom_decile"]

        self.assertEqual(strongest["symbols"], ["AAA", "BBB"])
        self.assertEqual(strongest["copy_text"], "AAA, BBB")
        self.assertTrue(strongest["details"][0].startswith("AAA"))
        self.assertIn("rs=+10.00", strongest["details"][0])
        self.assertEqual(weakest["symbols"], ["TTT", "SSS"])
        self.assertEqual(weakest["copy_text"], "SSS, TTT")
        self.assertTrue(weakest["details"][0].startswith("TTT"))
        self.assertIn("rs=-9.00", weakest["details"][0])

        report = format_market_prep_payload_report(payload)

        self.assertIn("Strongest Stocks: Top 10% D1 vs SPY", report)
        self.assertIn("Weakest Stocks: Bottom 10% D1 vs SPY", report)
        self.assertIn("Top 10% by D1 relative strength versus SPY (2 of 20", report)

    @staticmethod
    def _universe_frame(closes, *, start="2024-06-03"):
        """Build a normalized daily-bar frame from a list of closes."""
        closes = [float(c) for c in closes]
        dates = pd.bdate_range(start=start, periods=len(closes))
        return pd.DataFrame(
            {
                "datetime": dates,
                "open": closes,
                "high": [c * 1.001 for c in closes],
                "low": [c * 0.999 for c in closes],
                "close": closes,
                "volume": [1_000_000.0] * len(closes),
            }
        )

    @classmethod
    def _strength_frame(cls, strength_pct):
        """Frame whose 1d and 5d returns both equal ``strength_pct`` (so D1 RS vs a
        flat SPY benchmark equals ``strength_pct``)."""
        base = 100.0 / (1.0 + strength_pct / 100.0)
        return cls._universe_frame([base] * 7 + [100.0])

    def test_market_prep_decile_ranks_full_universe_not_just_flagged(self):
        symbols = [
            "AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ",
            "KKK", "LLL", "MMM", "NNN", "OOO", "PPP", "QQQ", "RRR", "SSS", "TTT",
        ]
        frames = {sym: self._strength_frame(10.0 - idx) for idx, sym in enumerate(symbols)}

        payload = build_market_prep_payload(
            daily_frames_by_symbol=frames,
            spy_benchmark={"one_day_return_pct": 0.0, "five_day_return_pct": 0.0},
            sides_by_symbol={sym: "LONG" for sym in symbols},
            priority_rows=[],  # nothing flagged -> proves the decile no longer needs flagged rows
            reference_date=date(2026, 4, 24),
            previous_session_date=date(2026, 4, 23),
            calendar_rows_by_date={},
        )
        sections = {section["id"]: section for section in payload["sections"]}

        self.assertEqual(sections["strongest_stocks_top_decile"]["symbols"], ["AAA", "BBB"])
        self.assertEqual(sections["weakest_stocks_bottom_decile"]["symbols"], ["TTT", "SSS"])
        self.assertIn("2 of 20 symbols with RS data", sections["strongest_stocks_top_decile"]["note"])

    def test_market_prep_decile_breaks_ties_by_symbol(self):
        # 20 symbols -> top decile = 2. "BBB" and "AAA" tie as the two strongest and are
        # inserted B-before-A; the tie must still resolve to alphabetical order.
        strengths = {"BBB": 10.0, "AAA": 10.0}
        for n in range(18):
            strengths[f"S{n:02d}"] = 1.0
        frames = {sym: self._strength_frame(val) for sym, val in strengths.items()}

        payload = build_market_prep_payload(
            daily_frames_by_symbol=frames,
            spy_benchmark={"one_day_return_pct": 0.0, "five_day_return_pct": 0.0},
            reference_date=date(2026, 4, 24),
            previous_session_date=date(2026, 4, 23),
            calendar_rows_by_date={},
        )
        sections = {section["id"]: section for section in payload["sections"]}
        self.assertEqual(sections["strongest_stocks_top_decile"]["symbols"], ["AAA", "BBB"])

    def test_market_prep_universe_skips_symbols_without_enough_history(self):
        frames = {
            "AAA": self._strength_frame(8.0),
            "SHORTY": self._universe_frame([100.0, 101.0]),  # only 2 bars -> no RS
        }
        payload = build_market_prep_payload(
            daily_frames_by_symbol=frames,
            spy_benchmark={"one_day_return_pct": 0.0, "five_day_return_pct": 0.0},
            reference_date=date(2026, 4, 24),
            previous_session_date=date(2026, 4, 23),
            calendar_rows_by_date={},
        )
        sections = {section["id"]: section for section in payload["sections"]}
        self.assertEqual(sections["strongest_stocks_top_decile"]["symbols"], ["AAA"])
        self.assertIn("1 of 1 symbols with RS data", sections["strongest_stocks_top_decile"]["note"])

    def test_market_prep_no_universe_data_falls_back_without_error(self):
        payload = build_market_prep_payload(
            priority_rows=[],
            reference_date=date(2026, 4, 24),
            previous_session_date=date(2026, 4, 23),
            calendar_rows_by_date={},
        )
        sections = {section["id"]: section for section in payload["sections"]}
        self.assertEqual(sections["strongest_stocks_top_decile"]["symbols"], [])
        self.assertEqual(sections["recently_strong_now_pulling_back"]["symbols"], [])

    def test_market_prep_pullback_lists_strong_then_pulling_back(self):
        # WINNER: up strongly over ~26w but down over the last 5 sessions.
        rising = [50.0 + i * 0.4 for i in range(135)]  # ~+100% over the window
        rising[-1] = rising[-6] * 0.95  # last 5 sessions negative
        # STEADY: flat the whole window (no pullback, not "recently strong").
        flat = [100.0] * 135

        frames = {
            "WINNER": self._universe_frame(rising),
            "STEADY": self._universe_frame(flat),
        }
        payload = build_market_prep_payload(
            daily_frames_by_symbol=frames,
            spy_benchmark={"one_day_return_pct": 0.0, "five_day_return_pct": 0.0},
            reference_date=date(2026, 4, 24),
            previous_session_date=date(2026, 4, 23),
            calendar_rows_by_date={},
        )
        sections = {section["id"]: section for section in payload["sections"]}
        pullback = sections["recently_strong_now_pulling_back"]
        self.assertIn("WINNER", pullback["symbols"])
        self.assertNotIn("STEADY", pullback["symbols"])
        self.assertTrue(pullback["details"][0].startswith("WINNER"))

    def test_symbol_industry_contexts_use_industry_map_then_sector_fallback(self):
        contexts = master_avwap.build_symbol_industry_contexts(
            ["NVDA", "JPM"],
            symbol_classification_cache={
                "NVDA": {
                    "sectorKey": "technology",
                    "industryKey": "semiconductors",
                    "sector": "Technology",
                    "industry": "Semiconductors",
                },
                "JPM": {
                    "sectorKey": "financial-services",
                    "industryKey": "banks-diversified",
                    "sector": "Financial Services",
                    "industry": "Banks - Diversified",
                },
            },
            sector_map={"technology": "XLK", "financial-services": "XLF"},
            industry_map_data={
                "yahoo_industryKey_to_ref": {
                    "semiconductors": {"etf": "SMH"},
                    "banks-diversified": {"etf": ""},
                }
            },
        )

        self.assertEqual(contexts["NVDA"]["industry_etf"], "SMH")
        self.assertEqual(contexts["NVDA"]["sector_etf"], "XLK")
        self.assertEqual(contexts["JPM"]["industry_etf"], "XLF")

    def test_universe_strength_rows_compute_stock_vs_industry(self):
        frames = {
            "NVDA": self._strength_frame(10.0),
        }
        industry_frames = {
            "SMH": self._strength_frame(3.0),
        }
        contexts = {
            "NVDA": {
                "sector": "Technology",
                "industry": "Semiconductors",
                "sector_etf": "XLK",
                "industry_etf": "SMH",
            }
        }

        rows = master_avwap.build_universe_strength_rows(
            frames,
            {"one_day_return_pct": 0.0, "five_day_return_pct": 0.0},
            sides_by_symbol={"NVDA": "LONG"},
            industry_context_by_symbol=contexts,
            industry_daily_frames_by_etf=industry_frames,
        )

        self.assertEqual(rows[0]["industry_etf"], "SMH")
        self.assertAlmostEqual(rows[0]["rs_vs_industry"], 7.0, places=3)
        self.assertAlmostEqual(rows[0]["industry_five_day_return_pct"], 3.0, places=3)

    def test_market_prep_includes_strongest_industries(self):
        industry_frames = {
            "XBI": self._strength_frame(2.0),
            "SMH": self._strength_frame(6.0),
        }
        payload = build_market_prep_payload(
            industry_daily_frames_by_etf=industry_frames,
            industry_context_by_symbol={
                "NVDA": {"industry": "Semiconductors", "industry_etf": "SMH"},
                "BIIB": {"industry": "Biotechnology", "industry_etf": "XBI"},
            },
            spy_benchmark={"one_day_return_pct": 0.0, "five_day_return_pct": 0.0},
            reference_date=date(2026, 4, 24),
            previous_session_date=date(2026, 4, 23),
            calendar_rows_by_date={},
        )
        sections = {section["id"]: section for section in payload["sections"]}

        self.assertEqual(sections["strongest_industries"]["symbols"][:2], ["SMH", "XBI"])
        self.assertIn("Semiconductors", sections["strongest_industries"]["details"][0])
        self.assertIn("rs=+6.00", sections["strongest_industries"]["details"][0])

    def test_industry_relative_strength_enrichment_is_feature_flag_gated(self):
        priority_rows = [
            {
                "symbol": "NVDA",
                "side": "LONG",
                "score": 100.0,
                "has_favorite_signal": True,
                "setup_family": "sma_breakout",
            }
        ]
        universe_rows = [
            {
                "symbol": "NVDA",
                "side": "LONG",
                "industry_etf": "SMH",
                "rs_vs_industry": 2.2,
                "industry_one_day_return_pct": 1.0,
                "industry_five_day_return_pct": 1.2,
            }
        ]
        industry_rows = [{"industry_etf": "SMH", "daily_relative_strength_score": 1.4}]

        master_avwap.enrich_priority_rows_with_industry_relative_strength(
            priority_rows,
            universe_rows,
            industry_rows,
            scoring_enabled=False,
        )
        self.assertEqual(priority_rows[0]["score"], 100.0)
        self.assertEqual(priority_rows[0]["industry_relative_strength_bonus"], 0)
        self.assertIn("score flag off", priority_rows[0]["industry_relative_strength_note"])

        master_avwap.enrich_priority_rows_with_industry_relative_strength(
            priority_rows,
            universe_rows,
            industry_rows,
            scoring_enabled=True,
        )
        self.assertEqual(priority_rows[0]["score"], 110.0)
        self.assertEqual(priority_rows[0]["industry_relative_strength_bonus"], 10)

    def test_industry_relative_strength_boost_skips_inactive_general_rows(self):
        priority_rows = [{"symbol": "NVDA", "side": "LONG", "score": 100.0, "setup_family": "general"}]
        universe_rows = [{"symbol": "NVDA", "side": "LONG", "industry_etf": "SMH", "rs_vs_industry": 2.2}]
        industry_rows = [{"industry_etf": "SMH", "daily_relative_strength_score": 1.4}]

        master_avwap.enrich_priority_rows_with_industry_relative_strength(
            priority_rows,
            universe_rows,
            industry_rows,
            scoring_enabled=True,
        )

        self.assertEqual(priority_rows[0]["score"], 100.0)
        self.assertEqual(priority_rows[0]["industry_relative_strength_bonus"], 0)
        self.assertIn("setup inactive", priority_rows[0]["industry_relative_strength_note"])

    def test_htf_trend_context_detects_aligned_sma_retest(self):
        intraday = _build_intraday_htf_retest_frame()

        context = master_avwap.assess_htf_trend_context(intraday, "LONG", scoring_enabled=False)

        self.assertEqual(context["htf_trend_1h"], "UP")
        self.assertEqual(context["htf_trend_4h"], "UP")
        self.assertTrue(context["htf_trend_aligned"])
        self.assertTrue(context["htf_retest_confirmed"])
        self.assertIn("4h:SMA_20", context["htf_retest_sma"])
        self.assertEqual(context["htf_trend_score_bonus"], 0)
        self.assertIn("score flag off", context["htf_trend_note"])

        scoring_context = master_avwap.assess_htf_trend_context(
            intraday,
            "LONG",
            scoring_enabled=True,
        )
        self.assertEqual(scoring_context["htf_trend_score_bonus"], master_avwap.HTF_TREND_SCORE_BONUS)

    def test_htf_trend_enrichment_is_feature_flag_gated_and_writes_study_rows(self):
        intraday = _build_intraday_htf_retest_frame()
        priority_rows = [
            {
                "symbol": "TSLA",
                "side": "LONG",
                "score": 100.0,
                "has_favorite_signal": True,
                "favorite_signals": ["BOUNCE_VWAP"],
                "context_signals": [],
                "setup_family": "avwap_bounce",
                "priority_bucket": "favorite_setup",
            }
        ]
        ai_state = {
            "symbols": {
                "TSLA": {
                    "symbol": "TSLA",
                    "side": "LONG",
                    "last_trade_date": "2026-06-01",
                    "last_close": 150.0,
                    "current_anchor": {"date": "2026-01-02", "vwap": 145.0, "bands": {"LOWER_1": 140.0}},
                    "setup_family": "avwap_bounce",
                }
            }
        }
        feature_rows_by_symbol = {"TSLA": {"priority_score": 100.0}}

        study_rows = master_avwap.enrich_priority_rows_with_htf_trend_context(
            priority_rows,
            intraday_frames_by_symbol={"TSLA": intraday},
            scoring_enabled=False,
            ai_state=ai_state,
            feature_rows_by_symbol=feature_rows_by_symbol,
        )

        self.assertEqual(priority_rows[0]["score"], 100.0)
        self.assertEqual(priority_rows[0]["htf_trend_score_bonus"], 0)
        self.assertEqual(len(study_rows), 1)
        self.assertEqual(study_rows[0]["setup_family"], master_avwap.HTF_TREND_STUDY_FAMILY)
        self.assertEqual(study_rows[0]["priority_bucket"], master_avwap.HTF_TREND_STUDY_BUCKET)
        self.assertEqual(feature_rows_by_symbol["TSLA"]["htf_trend_4h"], "UP")
        self.assertEqual(ai_state["symbols"]["TSLA"]["htf_retest_sma"], priority_rows[0]["htf_retest_sma"])

        boosted_rows = [dict(priority_rows[0], score=100.0, htf_trend_score_bonus=0)]
        boosted_features = {"TSLA": {"priority_score": 100.0}}
        master_avwap.enrich_priority_rows_with_htf_trend_context(
            boosted_rows,
            intraday_frames_by_symbol={"TSLA": intraday},
            scoring_enabled=True,
            ai_state={"symbols": {"TSLA": dict(ai_state["symbols"]["TSLA"])}},
            feature_rows_by_symbol=boosted_features,
        )

        self.assertEqual(boosted_rows[0]["htf_trend_score_bonus"], master_avwap.HTF_TREND_SCORE_BONUS)
        self.assertEqual(boosted_rows[0]["score"], 100.0 + master_avwap.HTF_TREND_SCORE_BONUS)
        self.assertEqual(boosted_features["TSLA"]["priority_score"], boosted_rows[0]["score"])

    def test_hv_level_enrichment_writes_study_rows_without_scoring(self):
        frame = _build_daily_bar_cache_frame(90)
        last_close = float(frame.iloc[-1]["close"])
        frame.loc[70, "high"] = last_close + 0.02
        frame.loc[70, "low"] = last_close - 0.02
        frame.loc[70, "close"] = last_close
        frame.loc[70, "volume"] = 5_000_000
        priority_rows = [
            {
                "symbol": "TSLA",
                "side": "LONG",
                "score": 100.0,
                "setup_family": "avwap_bounce",
            }
        ]
        ai_state = {
            "symbols": {
                "TSLA": {
                    "symbol": "TSLA",
                    "side": "LONG",
                    "last_trade_date": frame.iloc[-1]["datetime"].date().isoformat(),
                    "last_close": last_close,
                    "atr20": 2.0,
                    "current_anchor": {"date": "2026-01-02", "vwap": last_close, "bands": {"LOWER_1": last_close - 2.0}},
                }
            }
        }
        feature_rows_by_symbol = {"TSLA": {"last_close": last_close, "atr20": 2.0, "priority_score": 100.0}}

        with tempfile.TemporaryDirectory() as temp_dir:
            study_rows = master_avwap.enrich_priority_rows_with_hv_levels(
                priority_rows,
                {"TSLA": frame},
                ai_state=ai_state,
                feature_rows_by_symbol=feature_rows_by_symbol,
                levels_dir=Path(temp_dir),
                persist=False,
            )

        self.assertEqual(priority_rows[0]["score"], 100.0)
        self.assertGreaterEqual(priority_rows[0]["hv_level_nearby_count"], 1)
        self.assertEqual(len(study_rows), 1)
        self.assertEqual(study_rows[0]["setup_family"], master_avwap.HV_LEVEL_STUDY_FAMILY)
        self.assertEqual(study_rows[0]["priority_bucket"], master_avwap.HV_LEVEL_STUDY_BUCKET)
        self.assertIn("HV level", priority_rows[0]["hv_level_note"])
        self.assertEqual(feature_rows_by_symbol["TSLA"]["hv_level_nearby_count"], priority_rows[0]["hv_level_nearby_count"])
        self.assertEqual(ai_state["symbols"]["TSLA"]["hv_level_note"], priority_rows[0]["hv_level_note"])

    def test_hv_level_scoring_penalizes_entry_at_respected_level(self):
        # A respected green level just above a long entry should subtract score
        # only when scoring is enabled, scaled by cumulative respect history.
        def store(respect):
            return {
                "schema_version": 1,
                "symbol": "TSLA",
                "levels": [
                    {
                        "kind": "hv_horizontal",
                        "price": 100.10,
                        "bucket": "green",
                        "first_seen": "2024-01-01",
                        "respect_count": respect,
                        "break_count": 0,
                        "strength": 1.0,
                    }
                ],
            }

        off = master_avwap._hv_level_context_for_entry(
            store(8), side="LONG", entry_price=100.0, atr20=4.0,
            last_trade_date="2026-06-22", scoring_enabled=False,
        )
        self.assertEqual(off["hv_level_score_delta"], 0.0)

        fresh = master_avwap._hv_level_context_for_entry(
            store(0), side="LONG", entry_price=100.0, atr20=4.0,
            last_trade_date="2026-06-22", scoring_enabled=True,
        )
        respected = master_avwap._hv_level_context_for_entry(
            store(8), side="LONG", entry_price=100.0, atr20=4.0,
            last_trade_date="2026-06-22", scoring_enabled=True,
        )
        self.assertLess(fresh["hv_level_score_delta"], 0.0)
        self.assertLess(respected["hv_level_score_delta"], fresh["hv_level_score_delta"])

        # A level broken on the latest bar is a breakout, not a wall: no penalty.
        broken = store(8)
        broken["levels"][0]["last_break"] = "2026-06-22"
        breakout = master_avwap._hv_level_context_for_entry(
            broken, side="LONG", entry_price=100.0, atr20=4.0,
            last_trade_date="2026-06-22", scoring_enabled=True,
        )
        self.assertEqual(breakout["hv_level_score_delta"], 0.0)

    def test_compression_break_context_detects_close_out_of_prior_box(self):
        dates = pd.bdate_range("2026-01-01", periods=8)
        frame = pd.DataFrame(
            {
                "datetime": dates,
                "open": [100.0] * 8,
                "high": [100.4] * 7 + [102.0],
                "low": [99.6] * 7 + [101.0],
                "close": [100.0] * 7 + [101.2],
                "volume": [1000] * 8,
            }
        )

        context = master_avwap.assess_compression_break_context(
            frame,
            anchor_date_iso=dates[0].date().isoformat(),
            anchor_stdev=1.0,
            atr20=2.0,
            side="LONG",
            last_trade_date=dates[-1].date().isoformat(),
        )

        self.assertTrue(context["compression_break_today"])
        self.assertEqual(context["compression_break_direction"], "up")
        self.assertIn("Compression break up", context["compression_break_note"])

    def test_phase6_enrichment_writes_study_rows_without_scoring(self):
        dates = pd.bdate_range("2026-01-01", periods=8)
        last_trade_date = dates[-1].date().isoformat()
        frame = pd.DataFrame(
            {
                "datetime": dates,
                "open": [100.0] * 8,
                "high": [100.4] * 7 + [102.0],
                "low": [99.6] * 7 + [101.0],
                "close": [100.0] * 7 + [101.2],
                "volume": [1000] * 8,
            }
        )
        priority_rows = [
            {
                "symbol": "TSLA",
                "side": "LONG",
                "score": 100.0,
                "setup_family": "avwap_breakout",
                "trendline_break_recent": True,
                "trendline_break_note": "H-break 2026-01-02 -> 2026-01-06",
            }
        ]
        ai_state = {
            "symbols": {
                "TSLA": {
                    "symbol": "TSLA",
                    "side": "LONG",
                    "last_trade_date": last_trade_date,
                    "last_close": 101.2,
                    "atr20": 2.0,
                    "current_anchor": {"date": dates[0].date().isoformat(), "stdev": 1.0},
                }
            }
        }
        feature_rows_by_symbol = {"TSLA": {"last_close": 101.2, "atr20": 2.0, "last_trade_date": last_trade_date}}
        store = {
            "levels": [
                {
                    "kind": "cloud_flat",
                    "price": 101.2,
                    "bucket": "cloud",
                    "strength": 1.0,
                    "effective_range": [dates[-2].date().isoformat(), dates[-1].date().isoformat()],
                }
            ]
        }

        study_rows = master_avwap.enrich_priority_rows_with_phase6_studies(
            priority_rows,
            {"TSLA": frame},
            ai_state=ai_state,
            feature_rows_by_symbol=feature_rows_by_symbol,
            level_stores_by_symbol={"TSLA": store},
            persist=False,
        )

        self.assertEqual(priority_rows[0]["score"], 100.0)
        self.assertEqual(priority_rows[0]["cloud_level_nearby_count"], 1)
        self.assertTrue(priority_rows[0]["compression_break_today"])
        self.assertEqual(
            {row["setup_family"] for row in study_rows},
            {
                master_avwap.PHASE6_CLOUD_STUDY_FAMILY,
                master_avwap.PHASE6_COMPRESSION_BREAK_STUDY_FAMILY,
                master_avwap.PHASE6_TRENDLINE_BREAK_STUDY_FAMILY,
            },
        )
        self.assertEqual(feature_rows_by_symbol["TSLA"]["cloud_level_note"], priority_rows[0]["cloud_level_note"])
        self.assertEqual(ai_state["symbols"]["TSLA"]["compression_break_note"], priority_rows[0]["compression_break_note"])

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

    def test_load_or_refresh_earnings_merges_cached_calendar_rows_after_daily_flag(self):
        today = datetime.now().date()
        earnings_date = today - timedelta(days=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            earnings_cache_path = Path(temp_dir) / "earnings_dates_cache.json"
            calendar_cache_path = Path(temp_dir) / "earnings_calendar_rows.json"
            earnings_cache_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "last_recent_refresh_on": today.isoformat(),
                        "symbols": {
                            "XYZ": {
                                "dates": ["2026-02-26", "2025-11-06"],
                                "last_deep_refresh_on": "2026-02-24",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            calendar_cache_path.write_text(
                json.dumps(
                    {
                        earnings_date.isoformat(): {
                            "fetched_at": datetime.now().isoformat(timespec="seconds"),
                            "rows": [
                                {
                                    "symbol": "XYZ",
                                    "time": "time-after-hours",
                                    "name": "Block, Inc.",
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            old_calendar_cache = master_avwap._EARNINGS_CALENDAR_ROWS_CACHE
            master_avwap._EARNINGS_CALENDAR_ROWS_CACHE = None
            try:
                with patch.object(master_avwap, "EARNINGS_DATES_CACHE_FILE", earnings_cache_path), patch.object(
                    master_avwap,
                    "EARNINGS_CALENDAR_CACHE_FILE",
                    calendar_cache_path,
                ), patch("master_avwap.collect_earnings_dates", return_value={"XYZ": []}), patch(
                    "master_avwap.get_shared_past_earnings_dates",
                    return_value={},
                ), patch("master_avwap.record_shared_nasdaq_rows"):
                    loaded = master_avwap.load_or_refresh_earnings(["XYZ"])
            finally:
                master_avwap._EARNINGS_CALENDAR_ROWS_CACHE = old_calendar_cache

            self.assertEqual(loaded["XYZ"][0], earnings_date.isoformat())
            persisted = json.loads(earnings_cache_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["symbols"]["XYZ"]["dates"][0], earnings_date.isoformat())

    def test_load_or_refresh_earnings_merges_sec_recent_fallback_dates(self):
        today = datetime.now().date()
        recent_sec_date = (today - timedelta(days=2)).isoformat()
        prior_quarter = (today - timedelta(days=73)).isoformat()
        older_quarter = (today - timedelta(days=162)).isoformat()

        with tempfile.TemporaryDirectory() as temp_dir:
            earnings_cache_path = Path(temp_dir) / "earnings_dates_cache.json"
            earnings_cache_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "last_recent_refresh_on": today.isoformat(),
                        "symbols": {
                            "DJT": {
                                "dates": [prior_quarter, older_quarter],
                                "last_recent_refresh_on": today.isoformat(),
                                "last_deep_refresh_on": today.isoformat(),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(master_avwap, "EARNINGS_DATES_CACHE_FILE", earnings_cache_path), patch(
                "master_avwap.get_shared_past_earnings_dates",
                return_value={},
            ), patch(
                "master_avwap._collect_cached_calendar_earnings_dates",
                return_value={},
            ), patch(
                "master_avwap._collect_sec_recent_earnings_dates",
                return_value={"DJT": [recent_sec_date]},
            ):
                loaded = master_avwap.load_or_refresh_earnings(["DJT"])

        self.assertEqual(loaded["DJT"][0], recent_sec_date)
        self.assertEqual(loaded["DJT"][1], prior_quarter)

    def test_collect_sec_recent_earnings_dates_uses_recent_report_forms(self):
        today = date(2026, 5, 10)

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        def fake_get(url, headers=None, timeout=None):
            if url == master_avwap.SEC_COMPANY_TICKERS_URL:
                return FakeResponse({"0": {"ticker": "PBA", "cik_str": "1546066"}})
            return FakeResponse(
                {
                    "filings": {
                        "recent": {
                            "form": ["6-K", "8-K", "10-Q", "10-Q"],
                            "filingDate": ["2026-05-07", "2026-05-07", "2026-05-05", "2026-04-01"],
                        }
                    }
                }
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            earnings_cache_path = Path(temp_dir) / "earnings_dates_cache.json"
            with patch.object(master_avwap, "EARNINGS_DATES_CACHE_FILE", earnings_cache_path), patch(
                "master_avwap.requests.get",
                side_effect=fake_get,
            ):
                loaded = master_avwap._collect_sec_recent_earnings_dates(["PBA"], today=today)

        self.assertEqual(loaded["PBA"], ["2026-05-07", "2026-05-05"])

    def test_load_or_refresh_earnings_uses_shared_history_dates(self):
        today = datetime.now().date()
        shared_date = (today - timedelta(days=2)).isoformat()

        with tempfile.TemporaryDirectory() as temp_dir:
            earnings_cache_path = Path(temp_dir) / "earnings_dates_cache.json"
            earnings_cache_path.write_text(
                json.dumps({"schema_version": 2, "last_recent_refresh_on": today.isoformat(), "symbols": {}}),
                encoding="utf-8",
            )

            with patch.object(master_avwap, "EARNINGS_DATES_CACHE_FILE", earnings_cache_path), patch(
                "master_avwap.get_shared_past_earnings_dates",
                return_value={"XYZ": [shared_date]},
            ), patch(
                "master_avwap._collect_cached_calendar_earnings_dates",
                return_value={},
            ), patch(
                "master_avwap._can_attempt_yf_earnings_lookup",
                return_value=False,
            ), patch("master_avwap.collect_earnings_dates", return_value={"XYZ": []}):
                loaded = master_avwap.load_or_refresh_earnings(["XYZ"])

        self.assertEqual(loaded["XYZ"], [shared_date])

    def test_nasdaq_time_after_hours_infers_amc_release_session(self):
        self.assertEqual(
            master_avwap.infer_release_session({"symbol": "XYZ", "time": "time-after-hours"}),
            "amc",
        )

    def test_recent_fresh_earnings_shifts_previous_anchor_back_one_more_quarter(self):
        today = datetime.now().date()
        current_quarter = (today - timedelta(days=1)).isoformat()
        prior_quarter = (today - timedelta(days=80)).isoformat()
        older_quarter = (today - timedelta(days=170)).isoformat()

        dates = [current_quarter, prior_quarter, older_quarter]

        self.assertEqual(master_avwap.pick_current_earnings_anchor(dates).isoformat(), prior_quarter)
        self.assertEqual(master_avwap.pick_previous_earnings_anchor(dates).isoformat(), older_quarter)

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

    def test_release_context_anchors_post_earnings_avwap_to_day_before_gap_up(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=0)
        gap_idx = 30
        anchor_idx = gap_idx - 1
        gap_date = df.iloc[gap_idx]["datetime"].date().isoformat()
        anchor_date = df.iloc[anchor_idx]["datetime"].date().isoformat()

        context = _build_latest_earnings_release_context(
            df,
            {
                "earnings_date": gap_date,
                "release_session": "bmo",
            },
        )
        expected_vwap, expected_stdev, expected_bands = master_avwap.calc_anchored_vwap_bands(df, anchor_idx)

        self.assertTrue(context["active"])
        self.assertEqual(context["gap_idx"], gap_idx)
        self.assertEqual(context["anchor_idx"], anchor_idx)
        self.assertEqual(context["gap_date"], gap_date)
        self.assertEqual(context["anchor_date"], anchor_date)
        self.assertTrue(context["gap_is_up"])
        self.assertAlmostEqual(context["anchor_meta"]["vwap"], float(expected_vwap))
        self.assertAlmostEqual(context["anchor_meta"]["stdev"], float(expected_stdev))
        self.assertEqual(context["anchor_meta"]["bands"], {key: float(value) for key, value in expected_bands.items()})

    def test_release_context_records_inferred_unknown_session(self):
        df = _build_history_with_gap()
        earnings_date = df.iloc[20]["datetime"].date().isoformat()

        with patch("master_avwap.record_inferred_release_session") as record_mock:
            context = _build_latest_earnings_release_context(
                df,
                {
                    "symbol": "TEST",
                    "earnings_date": earnings_date,
                    "release_session": "unknown",
                },
            )

        self.assertTrue(context["active"])
        self.assertTrue(str(context["release_session"]).endswith("_inferred"))
        record_mock.assert_called_once()

    def test_post_earnings_52w_break_uses_earnings_candle_high_and_fresh_break(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=0)
        context = _build_post_earnings_52w_release_context(df)

        summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertTrue(summary["qualified_gap"])
        self.assertTrue(summary["qualified_52w_gap"])
        self.assertTrue(summary["gap_candle_directional"])
        self.assertEqual(summary["monitor_level"], 120.0)
        self.assertEqual(summary["monitor_level_label"], "52W_HIGH")
        self.assertEqual(summary["earnings_candle_stop_level"], 117.5)
        self.assertEqual(summary["earnings_candle_stop_label"], master_avwap.POST_EARNINGS_CANDLE_STOP_LABEL_LONG)
        self.assertTrue(summary["break_signal"])
        self.assertTrue(summary["break_fresh"])
        self.assertEqual(summary["break_age_sessions"], 0)
        self.assertEqual(summary["break_sessions_after_gap"], 1)
        self.assertIn(master_avwap.POST_EARNINGS_BREAK_SIGNAL, summary["events"])

    def test_post_earnings_52w_break_handles_short_52w_low_adbe_style(self):
        df = _build_post_earnings_52w_history(side="SHORT", extra_sessions_after_break=0)
        context = _build_post_earnings_52w_release_context(df, side="SHORT")

        summary = master_avwap.analyze_post_earnings_setups(df, "SHORT", context)

        self.assertTrue(summary["qualified_gap"])
        self.assertTrue(summary["qualified_52w_gap"])
        self.assertTrue(summary["gap_candle_directional"])
        self.assertEqual(summary["monitor_level"], 80.0)
        self.assertEqual(summary["monitor_level_label"], "52W_LOW")
        self.assertEqual(summary["earnings_candle_stop_level"], 82.0)
        self.assertEqual(summary["earnings_candle_stop_label"], master_avwap.POST_EARNINGS_CANDLE_STOP_LABEL_SHORT)
        self.assertTrue(summary["break_signal"])
        self.assertTrue(summary["break_fresh"])
        self.assertTrue(summary["break_close"])
        self.assertEqual(summary["family"], master_avwap.POST_EARNINGS_BREAK_SIGNAL)
        self.assertIn(master_avwap.POST_EARNINGS_BREAK_SIGNAL, summary["events"])
        self.assertIn(master_avwap.POST_EARNINGS_CLOSE_CONFIRM_SIGNAL, summary["events"])

    def test_post_earnings_52w_break_requires_green_gap_candle_for_longs(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=0)
        gap_idx = 30
        df.loc[df.index[gap_idx], "close"] = df.loc[df.index[gap_idx], "open"] - 0.25
        context = _build_post_earnings_52w_release_context(df)

        summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertFalse(summary["qualified_gap"])
        self.assertFalse(summary["qualified_52w_gap"])
        self.assertFalse(summary["break_signal"])
        self.assertEqual(summary["events"], [])

    def test_post_earnings_52w_break_can_start_on_day_after_earnings(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=1)
        df.loc[30, ["open", "high", "low", "close"]] = [119.05, 119.20, 117.50, 118.90]
        df.loc[32, ["open", "high", "low", "close"]] = [120.80, 122.00, 120.20, 121.60]
        context = _build_post_earnings_52w_release_context(df)

        summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertTrue(summary["qualified_gap"])
        self.assertEqual(summary["monitor_level"], 121.0)
        self.assertTrue(summary["break_signal"])
        self.assertEqual(summary["break_age_sessions"], 0)
        self.assertEqual(summary["first_break_date"], df.iloc[32]["datetime"].date().isoformat())
        self.assertIn("breakout candle", summary["note"])

    def test_post_earnings_52w_break_must_happen_within_three_candles(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=3)
        gap_idx = 30
        for idx in range(gap_idx + 1, gap_idx + 4):
            df.loc[df.index[idx], "high"] = 119.5
        df.loc[df.index[gap_idx + 4], "high"] = 121.0
        df.loc[df.index[gap_idx + 4], "close"] = 120.5
        context = _build_post_earnings_52w_release_context(df)

        summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertTrue(summary["qualified_52w_gap"])
        self.assertFalse(summary["break_signal"])
        self.assertFalse(summary["break_fresh"])
        self.assertEqual(summary["break_sessions_after_gap"], 4)
        self.assertNotIn(master_avwap.POST_EARNINGS_BREAK_SIGNAL, summary["events"])
        self.assertIn("outside the 1-3 setup window", summary["note"])

    def test_post_earnings_52w_day_after_breakout_waits_for_another_candle(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=0)
        df.loc[30, ["open", "high", "low", "close"]] = [119.05, 119.20, 117.50, 118.90]
        context = _build_post_earnings_52w_release_context(df)

        summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertTrue(summary["qualified_gap"])
        self.assertEqual(summary["monitor_level"], 121.0)
        self.assertFalse(summary["break_signal"])
        self.assertFalse(summary["break_intraday"])
        self.assertIsNone(summary["break_age_sessions"])

    def test_post_earnings_52w_breakout_after_day_after_does_not_qualify(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=1)
        df.loc[30, ["open", "high", "low", "close"]] = [119.05, 119.20, 117.50, 118.90]
        df.loc[31, ["open", "high", "low", "close"]] = [119.00, 119.25, 118.20, 119.00]
        df.loc[32, ["open", "high", "low", "close"]] = [120.80, 122.00, 120.20, 121.60]
        context = _build_post_earnings_52w_release_context(df)

        summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertFalse(summary["active"])
        self.assertFalse(summary["qualified_gap"])
        self.assertFalse(summary["break_signal"])
        self.assertEqual(summary["events"], [])

    def test_post_earnings_52w_break_goes_stale_after_two_candles(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=4)
        context = _build_post_earnings_52w_release_context(df)

        summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertTrue(summary["qualified_gap"])
        self.assertFalse(summary["break_signal"])
        self.assertFalse(summary["break_intraday"])
        self.assertFalse(summary["break_fresh"])
        self.assertEqual(summary["break_age_sessions"], 4)
        self.assertNotIn(master_avwap.POST_EARNINGS_BREAK_SIGNAL, summary["events"])
        self.assertIn("break stale", summary["note"])

    def test_post_earnings_setup_allows_through_10_market_days(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=9)
        context = _build_post_earnings_52w_release_context(df)
        self.assertEqual(context["sessions_since_gap"], 10)
        self.assertTrue(context["in_post_earnings_window"])

        summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertTrue(summary["active"])
        self.assertTrue(summary["qualified_gap"])
        self.assertFalse(summary["break_signal"])
        self.assertEqual(summary["events"], [])

    def test_post_earnings_setup_expires_after_10_market_days(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=10)
        context = _build_post_earnings_52w_release_context(df)
        self.assertEqual(context["sessions_since_gap"], 11)

        summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertFalse(summary["active"])
        self.assertFalse(summary["qualified_gap"])
        self.assertEqual(summary["events"], [])

    def test_post_earnings_avwape_bounce_does_not_require_52w_gap(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=4)
        df.loc[:29, "high"] = 130.0
        context = _build_post_earnings_52w_release_context(df)

        with patch("master_avwap.bounce_up_at_level", return_value=True) as bounce_mock:
            summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        bounce_mock.assert_called()
        self.assertTrue(summary["active"])
        self.assertTrue(summary["qualified_gap"])
        self.assertTrue(summary["qualified_pre_earnings_avwap_gap"])
        self.assertFalse(summary["qualified_52w_gap"])
        self.assertFalse(summary["break_signal"])
        self.assertTrue(summary["bounce_signal"])
        self.assertEqual(summary["monitor_level"], 115.0)
        self.assertEqual(summary["monitor_level_label"], "PRE_EARN_AVWAP")
        self.assertEqual(summary["events"], [master_avwap.POST_EARNINGS_BOUNCE_SIGNAL])

    def test_post_earnings_avwape_rejection_allows_short_pierce_and_close_below_adbe_style(self):
        df = _build_post_earnings_52w_history(side="SHORT", extra_sessions_after_break=4)
        df.loc[:29, "low"] = 70.0
        df.loc[df.index[-1], ["open", "high", "low", "close"]] = [84.8, 86.5, 83.8, 84.2]
        context = _build_post_earnings_52w_release_context(df, side="SHORT")

        self.assertFalse(master_avwap.bounce_down_at_level(df, 85.0))

        summary = master_avwap.analyze_post_earnings_setups(df, "SHORT", context)

        self.assertTrue(summary["active"])
        self.assertTrue(summary["qualified_gap"])
        self.assertTrue(summary["qualified_pre_earnings_avwap_gap"])
        self.assertFalse(summary["qualified_52w_gap"])
        self.assertFalse(summary["break_signal"])
        self.assertTrue(summary["bounce_signal"])
        self.assertEqual(summary["monitor_level"], 85.0)
        self.assertEqual(summary["monitor_level_label"], "PRE_EARN_AVWAP")
        self.assertEqual(summary["family"], master_avwap.POST_EARNINGS_BOUNCE_SIGNAL)
        self.assertEqual(summary["events"], [master_avwap.POST_EARNINGS_BOUNCE_SIGNAL])
        self.assertIn("pre-earnings AVWAPE rejection", summary["note"])

    def test_post_earnings_avwape_bounce_can_fire_within_10_day_window(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=4)
        context = _build_post_earnings_52w_release_context(df)
        self.assertEqual(context["sessions_since_gap"], 5)
        df.loc[df.index[-1], "high"] = 119.5
        df.loc[df.index[-1], "close"] = 119.4

        with patch("master_avwap.bounce_up_at_level", return_value=True):
            summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertTrue(summary["active"])
        self.assertTrue(summary["qualified_gap"])
        self.assertFalse(summary["break_signal"])
        self.assertTrue(summary["bounce_signal"])
        self.assertEqual(summary["bounce_age_sessions"], 0)
        self.assertEqual(summary["bounce_date"], df.iloc[-1]["datetime"].date().isoformat())
        self.assertEqual(summary["family"], master_avwap.POST_EARNINGS_BOUNCE_SIGNAL)
        self.assertEqual(summary["events"], [master_avwap.POST_EARNINGS_BOUNCE_SIGNAL])

    def test_post_earnings_avwape_bounce_waits_until_second_session_after_gap(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=0)
        df.loc[:29, "high"] = 130.0
        context = _build_post_earnings_52w_release_context(df)
        self.assertEqual(context["sessions_since_gap"], 1)

        with patch("master_avwap.bounce_up_at_level", return_value=True) as bounce_mock:
            summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        bounce_mock.assert_not_called()
        self.assertTrue(summary["active"])
        self.assertFalse(summary["bounce_signal"])
        self.assertEqual(summary["events"], [])

    def test_post_earnings_avwape_bounce_can_fire_on_second_session_after_gap(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=1)
        df.loc[:29, "high"] = 130.0
        context = _build_post_earnings_52w_release_context(df)
        self.assertEqual(context["sessions_since_gap"], 2)

        with patch("master_avwap.bounce_up_at_level", return_value=True):
            summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertTrue(summary["bounce_signal"])
        self.assertEqual(summary["bounce_sessions_after_gap"], 2)
        self.assertEqual(summary["events"], [master_avwap.POST_EARNINGS_BOUNCE_SIGNAL])

    def test_post_earnings_52w_break_rejects_gap_day_break_and_bounce(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=0).iloc[:-1].copy()
        context = _build_post_earnings_52w_release_context(df)
        self.assertEqual(context["sessions_since_gap"], 0)

        with patch("master_avwap.bounce_up_at_level", return_value=True) as bounce_mock:
            summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        bounce_mock.assert_not_called()
        self.assertTrue(summary["active"])
        self.assertTrue(summary["qualified_gap"])
        self.assertFalse(summary["break_signal"])
        self.assertFalse(summary["break_fresh"])
        self.assertIsNone(summary["break_age_sessions"])
        self.assertEqual(summary["first_break_date"], "")
        self.assertFalse(summary["bounce_signal"])
        self.assertEqual(summary["events"], [])
        self.assertIn("waiting for post-gap break", summary["note"])

    def test_post_earnings_avwape_bounce_stays_flagged_for_two_sessions(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=4)
        context = _build_post_earnings_52w_release_context(df)
        two_sessions_ago_len = len(df) - 2

        def _bounce_for_slice(slice_df, _level):
            return len(slice_df) == two_sessions_ago_len

        with patch("master_avwap.bounce_up_at_level", side_effect=_bounce_for_slice):
            summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertFalse(summary["break_signal"])
        self.assertTrue(summary["bounce_signal"])
        self.assertEqual(summary["bounce_age_sessions"], 2)
        self.assertEqual(summary["events"], [master_avwap.POST_EARNINGS_BOUNCE_SIGNAL])

    def test_post_earnings_avwape_bounce_expires_after_two_sessions(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=4)
        context = _build_post_earnings_52w_release_context(df)
        three_sessions_ago_len = len(df) - 3

        def _bounce_for_slice(slice_df, _level):
            return len(slice_df) == three_sessions_ago_len

        with patch("master_avwap.bounce_up_at_level", side_effect=_bounce_for_slice):
            summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertFalse(summary["break_signal"])
        self.assertFalse(summary["bounce_signal"])
        self.assertEqual(summary["events"], [])
        self.assertIn("no AVWAPE bounce in last 2 session", summary["note"])

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
                "post_earnings_earnings_candle_stop_level": 54.0,
                "post_earnings_earnings_candle_stop_label": master_avwap.POST_EARNINGS_CANDLE_STOP_LABEL_SHORT,
                "ema_15": 52.0,
                "entry_feature_snapshot": {"ema8": 50.5},
                "atr20": 2.0,
            },
        )

        post_stop = next(item for item in candidates if item["label"] == POST_EARNINGS_STOP_LABEL)
        candle_stop = next(item for item in candidates if item["label"] == master_avwap.POST_EARNINGS_CANDLE_STOP_LABEL_SHORT)
        ema15_stop = next(item for item in candidates if item["label"] == "EMA_15")
        ema8_stop = next(item for item in candidates if item["label"] == "EMA_8")
        self.assertEqual(post_stop["source_type"], "post_earnings_anchor")
        self.assertEqual(post_stop["level"], 51.5)
        self.assertEqual(post_stop["close_failure_limit"], POST_EARNINGS_STOP_FAILURE_CLOSES)
        self.assertEqual(candle_stop["source_type"], "post_earnings_candle")
        self.assertEqual(candle_stop["level"], 54.0)
        self.assertEqual(candle_stop["close_failure_limit"], POST_EARNINGS_STOP_FAILURE_CLOSES)
        self.assertEqual(ema15_stop["source_type"], "ema")
        self.assertEqual(ema15_stop["level"], 52.0)
        self.assertEqual(ema15_stop["close_failure_limit"], POST_EARNINGS_STOP_FAILURE_CLOSES)
        self.assertEqual(ema8_stop["source_type"], "ema")
        self.assertEqual(ema8_stop["level"], 50.5)
        self.assertEqual(ema8_stop["close_failure_limit"], POST_EARNINGS_STOP_FAILURE_CLOSES)

    def test_post_earnings_tracker_record_builds_ema_stops_from_indicator_snapshot(self):
        setup = master_avwap.build_tracker_setup_record(
            {
                "symbol": "NVDA",
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "setup_family": "post_earnings_52w_break",
                "post_earnings_earnings_candle_stop_level": 94.0,
                "post_earnings_earnings_candle_stop_label": master_avwap.POST_EARNINGS_CANDLE_STOP_LABEL_LONG,
            },
            {
                "last_close": 100.0,
                "current_anchor": {
                    "date": "2026-05-01",
                    "vwap": 98.0,
                    "bands": {"LOWER_1": 96.0, "UPPER_1": 104.0, "UPPER_2": 108.0, "UPPER_3": 112.0},
                },
                "post_earnings_anchor": {
                    "vwap": 97.0,
                    "bands": {"LOWER_1": 95.0, "UPPER_1": 105.0, "UPPER_2": 110.0, "UPPER_3": 115.0},
                },
                "atr20": 2.0,
            },
            feature_row=None,
            generated_at="2026-05-05T13:00:00",
            indicator_row=pd.Series({"ema_8": 98.5, "ema_15": 96.5, "ema_21": 95.5, "atr_20": 2.0}),
            scan_date="2026-05-05",
        )

        stop_labels = {
            scenario["stop_reference_label"]
            for scenario in setup["scenarios"].values()
            if scenario.get("stop_reference_label")
        }
        self.assertIn("EMA_15", stop_labels)
        self.assertIn("EMA_8", stop_labels)

    def test_post_earnings_tracker_targets_can_use_pre_earnings_anchor_overrides(self):
        scenario = {
            "tradeable": True,
            "entry_price": 100.0,
            "initial_risk_per_share": 5.0,
            "initial_risk_usd": 500.0,
            "direction": 1.0,
            "shares": 100,
            "remaining_shares": 100,
            "status": "OPEN",
            "events": [],
            "partial_taken": False,
            "partial_shares": 0,
            "realized_pnl": 0.0,
            "realized_r": 0.0,
            "unrealized_pnl": 0.0,
            "unrealized_r": 0.0,
            "total_pnl": 0.0,
            "total_r": 0.0,
            "close_failure_count": 0,
            "stop_reference_label": POST_EARNINGS_STOP_LABEL,
            "active_stop_label": POST_EARNINGS_STOP_LABEL,
            "final_target_label": "UPPER_2",
            "partial_target_label": None,
            "trail_after_partial_label": None,
        }

        events = master_avwap._evaluate_tracker_scenario_bar(
            scenario,
            "LONG",
            "2026-05-07",
            pd.Series({"open": 103.0, "high": 110.5, "low": 102.0, "close": 108.0}),
            {"bands": {"UPPER_2": 200.0}, "vwap": 95.0},
            None,
            is_entry_day=False,
            dynamic_level_overrides={
                POST_EARNINGS_STOP_LABEL: 95.0,
                "UPPER_2": 110.0,
            },
        )

        self.assertEqual(scenario["status"], "TARGET_HIT")
        self.assertEqual(events[0]["reason"], "FINAL_TARGET")
        self.assertEqual(events[0]["price"], 110.0)

    def test_post_earnings_stop_candidates_include_ema15_add_reference(self):
        candidates = _find_tracker_stop_candidates(
            {
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "setup_family": "post_earnings_52w_break",
            },
            {
                "current_anchor": {
                    "vwap": 48.0,
                    "bands": {"LOWER_1": 46.0},
                },
                "post_earnings_anchor": {
                    "vwap": 47.5,
                    "bands": {"LOWER_1": 45.0},
                },
                "ema_15": 49.25,
                "atr20": 2.0,
            },
        )

        ema15_stop = next(item for item in candidates if item["label"] == "EMA_15")
        self.assertEqual(ema15_stop["source_type"], "ema")
        self.assertEqual(ema15_stop["level"], 49.25)
        self.assertEqual(ema15_stop["close_failure_limit"], POST_EARNINGS_STOP_FAILURE_CLOSES)

    def test_post_earnings_attributes_flag_ema15_pullback_ready(self):
        attributes, _ = master_avwap.build_tracker_entry_attributes(
            {
                "side": "LONG",
                "symbol": "AAPL",
                "priority_bucket": "favorite_setup",
                "post_earnings_active": True,
            },
            {"side": "LONG", "symbol": "AAPL"},
            {
                "directional_ema15_distance_atr": 0.25,
                "directional_ema15_bucket": "TIGHT_TO_EMA15",
                "ema15_reclaim": True,
            },
        )

        self.assertTrue(attributes["pattern.post_earnings_ema15_pullback_ready"])
        self.assertEqual(attributes["structure.directional_ema15_bucket"], "TIGHT_TO_EMA15")

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

    def test_tracker_family_stats_alias_legacy_favorite_zone_to_avwape_to_first_dev(self):
        reference_date = date(2026, 5, 12)
        setups = {
            "legacy_1": _build_tracker_setup(
                "NOK",
                "2026-05-08",
                1.2,
                setup_family="avwap_breakout",
                favorite_zone="AVWAPE to UPPER_1",
                favorite_signals=["CROSS_UP_UPPER_1"],
            ),
            "legacy_2": _build_tracker_setup(
                "SNDK",
                "2026-05-07",
                1.5,
                setup_family="favorite_zone_watch",
                favorite_zone="AVWAPE to UPPER_1",
            ),
        }

        recent_rows = build_recent_tracker_setup_family_rows(setups, reference_date=reference_date)
        family_names = {row["setup_family"] for row in recent_rows}

        self.assertIn(master_avwap.AVWAPE_TO_FIRST_DEV_FAMILY, family_names)
        self.assertNotIn("avwap_breakout", family_names)
        self.assertNotIn("favorite_zone_watch", family_names)

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

    def test_tracker_outcome_summary_clips_extreme_r_for_scoring(self):
        setup = _build_tracker_setup("JBL", "2026-03-31", 922.36)

        outcome = master_avwap._summarize_tracker_setup_outcome(setup)

        self.assertEqual(outcome["avg_closed_r"], master_avwap.TRACKER_SCORING_R_CLIP)
        self.assertEqual(outcome["raw_avg_closed_r"], 922.36)
        self.assertTrue(outcome["outlier_clipped"])

    def test_tracker_scenario_rejects_tiny_risk_floor(self):
        scenario = master_avwap._build_tracker_scenario(
            100.0,
            {"label": "AVWAPE", "level": 99.99, "source_type": "current_anchor"},
            "LONG",
            "favorite_setup",
            {"id": "full_band2", "label": "Full at band 2"},
        )

        self.assertFalse(scenario["tradeable"])
        self.assertEqual(scenario["shares"], 0)
        self.assertGreaterEqual(scenario["risk_floor_per_share"], 0.10)
        self.assertIn("below tracker floor", scenario["inactive_reason"])

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

    def test_setup_type_adjustment_uses_legacy_favorite_zone_stats_for_new_family(self):
        tracker_payload = {
            "setup_type_stats": [
                _build_setup_type_stat(
                    side="LONG",
                    favorite_zone="AVWAPE to UPPER_1",
                    retest_label="None",
                    setup_family="avwap_breakout",
                    tracked_setups=30,
                    closed_setups=16,
                    avg_closed_r=1.10,
                    baseline_avg_closed_r=0.15,
                    target_hit_rate=0.70,
                    baseline_target_hit_rate=0.45,
                    stop_rate=0.30,
                    baseline_stop_rate=0.55,
                ),
            ]
        }
        priority_rows = [
            {
                "symbol": "NOK",
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "setup_family": master_avwap.AVWAPE_TO_FIRST_DEV_FAMILY,
                "favorite_zone": "AVWAPE to UPPER_1",
                "retest_followthrough": False,
                "retest_reference_level": "",
                "compression_flag": False,
                "score": 100.0,
            },
        ]
        ai_state = {"symbols": {"NOK": {}}}
        feature_rows_by_symbol = {"NOK": {"priority_score": 100.0}}

        apply_tracker_setup_type_adjustments(
            priority_rows,
            ai_state,
            feature_rows_by_symbol,
            tracker_payload=tracker_payload,
        )

        self.assertGreater(priority_rows[0]["score"], 100.0)
        self.assertIn("setup type rank", priority_rows[0]["setup_type_score_note"])

    def test_live_scoring_config_signal_weights_are_capped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "scoring.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "signal_weights": {
                            "current": {
                                "SHORT": {
                                    "CROSS_DOWN_LOWER_2": 327,
                                    "POST_EARNINGS_52W_BREAK": 209,
                                }
                            },
                            "context": {"SHORT": {"PREV_CROSS_DOWN_LOWER_2": 99}},
                        },
                        "attribute_adjustments": [],
                        "feature_flags": {
                            "industry_relative_strength_scoring_enabled": True,
                            "htf_trend_scoring_enabled": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(master_avwap, "SCORING_CONFIG_FILE", config_path):
                master_avwap._PRIORITY_SCORING_CONFIG_CACHE = None
                config = master_avwap.load_priority_scoring_config(force_reload=True)

        master_avwap._PRIORITY_SCORING_CONFIG_CACHE = None
        self.assertEqual(config["signal_weights"]["current"]["SHORT"]["CROSS_DOWN_LOWER_2"], 110)
        self.assertEqual(config["signal_weights"]["current"]["SHORT"]["POST_EARNINGS_52W_BREAK"], 120)
        self.assertEqual(config["signal_weights"]["context"]["SHORT"]["PREV_CROSS_DOWN_LOWER_2"], 28)
        self.assertTrue(config["feature_flags"]["industry_relative_strength_scoring_enabled"])
        self.assertTrue(config["feature_flags"]["htf_trend_scoring_enabled"])
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

        self.assertEqual(priority_rows[0]["market_regime_score_delta"], -30)
        self.assertEqual(priority_rows[0]["score"], 70.0)
        self.assertEqual(feature_rows_by_symbol["TSLA"]["priority_score"], 70.0)

    def test_tracker_guardrails_make_unconfirmed_short_mid_earnings_ema_watch_only(self):
        priority_rows = [
            {
                "symbol": "TSLA",
                "side": "SHORT",
                "score": 180.0,
                "setup_family": "mid_earnings_ema15_retest",
                "favorite_signals": [master_avwap.MID_EARNINGS_EMA15_RETEST_SIGNAL],
                "context_signals": [],
                "has_favorite_signal": True,
                "trend_20d": "UP",
            }
        ]
        ai_state = {"symbols": {"TSLA": {"trend_20d": "UP"}}}
        feature_rows_by_symbol = {"TSLA": {"priority_score": 180.0}}

        apply_tracker_scoring_guardrails(priority_rows, ai_state, feature_rows_by_symbol)
        master_avwap.apply_final_priority_buckets(priority_rows, ai_state, [], feature_rows_by_symbol)

        self.assertEqual(priority_rows[0]["score"], master_avwap.PRIORITY_SHORT_MID_EARNINGS_EMA_WATCH_ONLY_CAP)
        self.assertTrue(priority_rows[0]["watch_only"])
        self.assertEqual(priority_rows[0]["priority_bucket"], "near_favorite_zone")
        self.assertIn("capped", priority_rows[0]["tracker_guardrail_score_note"])
        self.assertEqual(feature_rows_by_symbol["TSLA"]["priority_score"], priority_rows[0]["score"])

    def test_tracker_guardrails_boost_confirmed_short_first_dev_and_long_ema15(self):
        priority_rows = [
            {
                "symbol": "TSLA",
                "side": "SHORT",
                "score": 184.0,
                "setup_family": "mid_earnings_1stdev_retest",
                "favorite_signals": [master_avwap.MID_EARNINGS_FIRST_DEV_RETEST_SIGNAL],
                "context_signals": [],
                "has_favorite_signal": True,
                "trend_20d": "DOWN",
            },
            {
                "symbol": "AAPL",
                "side": "LONG",
                "score": 135.0,
                "setup_family": "mid_earnings_ema15_retest",
                "favorite_signals": [master_avwap.MID_EARNINGS_EMA15_RETEST_SIGNAL],
                "context_signals": [],
                "has_favorite_signal": True,
                "trend_20d": "UP",
            },
        ]
        ai_state = {"symbols": {"TSLA": {"trend_20d": "DOWN"}, "AAPL": {"trend_20d": "UP"}}}
        feature_rows_by_symbol = {"TSLA": {"priority_score": 184.0}, "AAPL": {"priority_score": 135.0}}

        apply_tracker_scoring_guardrails(priority_rows, ai_state, feature_rows_by_symbol)

        self.assertEqual(priority_rows[0]["score"], 184.0 + master_avwap.PRIORITY_SHORT_MID_EARNINGS_FIRST_DEV_SCORE_BONUS)
        self.assertIn("1st-dev confirmation", priority_rows[0]["tracker_guardrail_score_note"])
        self.assertEqual(priority_rows[1]["score"], 135.0 + master_avwap.PRIORITY_LONG_MID_EARNINGS_EMA15_SCORE_BONUS)
        self.assertIn("EMA15 retest tracker bonus", priority_rows[1]["tracker_guardrail_score_note"])

    def test_short_clean_first_zone_no_longer_gets_auto_favorite_bonus(self):
        priority_rows = [
            {
                "symbol": "TSLA",
                "side": "SHORT",
                "score": 98.0,
                "favorite_signals": [],
                "context_signals": [],
                "has_favorite_signal": False,
                "favorite_zone": "LOWER_1 to AVWAPE",
                "current_active_level": "VWAP",
                "current_band_zone": "VWAP to LOWER_1",
            }
        ]
        ai_state = {"symbols": {"TSLA": {}}}
        feature_rows_by_symbol = {"TSLA": {"priority_score": 98.0}}

        master_avwap.apply_clean_first_zone_score_bonus(priority_rows, ai_state, feature_rows_by_symbol)
        apply_tracker_scoring_guardrails(priority_rows, ai_state, feature_rows_by_symbol)
        master_avwap.apply_final_priority_buckets(priority_rows, ai_state, [], feature_rows_by_symbol)

        self.assertEqual(priority_rows[0]["clean_first_zone_score_bonus"], 0)
        self.assertEqual(priority_rows[0]["score"], master_avwap.PRIORITY_SHORT_NEAR_FAVORITE_UNCONFIRMED_CAP)
        self.assertEqual(priority_rows[0]["priority_bucket"], "near_favorite_zone")
        self.assertTrue(priority_rows[0]["watch_only"])

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
        self.assertEqual(priority_rows[1]["rejection_score_cap"], 160.0)
        self.assertEqual(priority_rows[1]["score"], 160.0)

    def test_rejection_cap_allows_trade_ready_mid_earnings_first_dev_retest(self):
        priority_rows = [
            {
                "symbol": "LAUR",
                "side": "LONG",
                "setup_family": master_avwap.MID_EARNINGS_EMA15_RETEST_FAMILY,
                "score": 204.0,
                "first_dev_chop_penalty": master_avwap.PRIORITY_FIRST_DEV_CHOP_PENALTY,
                "mid_earnings_ema15_trigger": True,
                "mid_earnings_first_dev_trigger": True,
                "previous_day_range_break": True,
                "current_band_zone": "UPPER_1 to UPPER_2",
                "current_day_open": 36.10,
                "last_close": 36.43,
                "intraday_vwap": 36.12,
            },
            {
                "symbol": "TGT",
                "side": "LONG",
                "setup_family": master_avwap.MID_EARNINGS_EMA15_RETEST_FAMILY,
                "score": 452.0,
                "first_dev_chop_penalty": master_avwap.PRIORITY_FIRST_DEV_CHOP_PENALTY,
                "mid_earnings_ema15_trigger": True,
                "mid_earnings_first_dev_trigger": True,
                "previous_day_range_break": False,
                "current_band_zone": "UPPER_1 to UPPER_2",
                "current_day_open": 129.0,
                "last_close": 130.7,
                "intraday_vwap": 130.5,
            },
        ]
        ai_state = {"symbols": {"LAUR": {}, "TGT": {}}}
        feature_rows_by_symbol = {
            "LAUR": {"priority_score": 204.0},
            "TGT": {"priority_score": 452.0},
        }

        master_avwap.apply_priority_rejection_score_caps(
            priority_rows,
            ai_state,
            feature_rows_by_symbol,
        )

        self.assertIsNone(priority_rows[0]["rejection_score_cap"])
        self.assertEqual(priority_rows[0]["score"], 204.0)
        self.assertEqual(priority_rows[1]["rejection_score_cap"], master_avwap.PRIORITY_REJECTION_CAP_FIRST_DEV_CHOP)
        self.assertEqual(priority_rows[1]["score"], master_avwap.PRIORITY_REJECTION_CAP_FIRST_DEV_CHOP)
        self.assertIn("first-dev chop/failure", priority_rows[1]["rejection_score_cap_note"])

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

    def test_pre_earnings_priority_block_uses_less_than_ten_day_boundary(self):
        blocked = master_avwap._next_earnings_window_summary(
            date(2026, 5, 5),
            ["2026-05-14"],
        )
        allowed = master_avwap._next_earnings_window_summary(
            date(2026, 5, 5),
            ["2026-05-15"],
        )

        self.assertTrue(blocked["pre_earnings_setup_blocked"])
        self.assertEqual(blocked["days_to_next_earnings"], 9)
        self.assertFalse(allowed["pre_earnings_setup_blocked"])
        self.assertEqual(allowed["days_to_next_earnings"], 10)

    def test_fetch_earnings_for_date_treats_null_data_as_empty_calendar(self):
        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": None}

        cache = {}
        future_date = (datetime.now().date() + timedelta(days=30)).isoformat()
        with (
            patch.object(master_avwap, "_load_earnings_calendar_rows_cache", return_value=cache),
            patch.object(master_avwap, "_save_earnings_calendar_rows_cache") as save_cache,
            patch.object(master_avwap.requests, "get", return_value=_Response()),
            patch.object(master_avwap.time, "sleep") as sleep_mock,
        ):
            rows = master_avwap.fetch_earnings_for_date(future_date)

        self.assertEqual(rows, [])
        self.assertEqual(cache[future_date]["rows"], [])
        save_cache.assert_called_once()
        sleep_mock.assert_not_called()

    def test_pre_earnings_priority_block_removes_non_theta_recommendation_bucket(self):
        priority_rows = [
            {
                "symbol": "AAPL",
                "side": "LONG",
                "score": 120,
                "favorite_signals": ["CROSS_UP_VWAP"],
                "context_signals": [],
                "has_favorite_signal": True,
                "favorite_zone": "AVWAPE to UPPER_1",
                "next_earnings_date": "2026-05-14",
                "days_to_next_earnings": 9,
            },
            {
                "symbol": "MSFT",
                "side": "LONG",
                "score": 118,
                "favorite_signals": ["CROSS_UP_VWAP"],
                "context_signals": [],
                "has_favorite_signal": True,
                "favorite_zone": "AVWAPE to UPPER_1",
                "next_earnings_date": "2026-05-15",
                "days_to_next_earnings": 10,
            },
        ]
        ai_state = {"symbols": {"AAPL": {}, "MSFT": {}}}
        feature_rows_by_symbol = {"AAPL": {}, "MSFT": {}}

        master_avwap.apply_pre_earnings_priority_blocks(
            priority_rows,
            ai_state,
            feature_rows_by_symbol,
        )
        master_avwap.apply_final_priority_buckets(
            priority_rows,
            ai_state,
            [],
            feature_rows_by_symbol,
        )
        attach_setup_candidate_payloads(priority_rows, ai_state, feature_rows_by_symbol)

        self.assertTrue(priority_rows[0]["ranking_blocked"])
        self.assertTrue(priority_rows[0]["pre_earnings_setup_blocked"])
        self.assertEqual(priority_rows[0]["priority_bucket"], "")
        self.assertIn("non-theta setup blocked", priority_rows[0]["candidate_rejection_reasons"][0])
        self.assertFalse(priority_rows[1]["pre_earnings_setup_blocked"])
        self.assertEqual(priority_rows[1]["priority_bucket"], "favorite_setup")

    def test_final_priority_bucket_demotes_low_score_favorite_signal(self):
        priority_rows = [
            {
                "symbol": "SEDG",
                "side": "LONG",
                "score": 39,
                "favorite_signals": ["CROSS_UP_VWAP"],
                "context_signals": ["PREV_CROSS_UP_VWAP"],
                "has_favorite_signal": True,
                "favorite_zone": "AVWAPE to UPPER_1",
            }
        ]
        ai_state = {"symbols": {"SEDG": {"favorite_zone": "AVWAPE to UPPER_1"}}}
        feature_rows_by_symbol = {"SEDG": {}}

        master_avwap.apply_final_priority_buckets(
            priority_rows,
            ai_state,
            [],
            feature_rows_by_symbol,
        )

        self.assertEqual(priority_rows[0]["priority_bucket"], "near_favorite_zone")
        self.assertFalse(priority_rows[0]["is_favorite_setup"])
        self.assertTrue(priority_rows[0]["is_near_favorite_zone"])
        self.assertIn("demoted from favorite", priority_rows[0]["favorite_score_gate_note"])
        self.assertEqual(feature_rows_by_symbol["SEDG"]["priority_bucket"], "near_favorite_zone")

    def test_final_priority_bucket_keeps_only_main_swing_setups_as_favorites(self):
        priority_rows = [
            {
                "symbol": "NOK",
                "side": "LONG",
                "score": 125,
                "setup_family": master_avwap.AVWAPE_TO_FIRST_DEV_FAMILY,
                "favorite_signals": ["CROSS_UP_UPPER_1"],
                "context_signals": [],
                "has_favorite_signal": True,
                "favorite_zone": "AVWAPE to UPPER_1",
            },
            {
                "symbol": "CHOP",
                "side": "LONG",
                "score": 180,
                "setup_family": "extreme_move_retest",
                "favorite_signals": ["EXTREME_MOVE_RETEST"],
                "context_signals": [],
                "has_favorite_signal": True,
            },
        ]
        ai_state = {"symbols": {"NOK": {}, "CHOP": {}}}
        feature_rows_by_symbol = {"NOK": {}, "CHOP": {}}

        master_avwap.apply_final_priority_buckets(
            priority_rows,
            ai_state,
            [],
            feature_rows_by_symbol,
        )

        self.assertEqual(priority_rows[0]["priority_bucket"], "favorite_setup")
        self.assertEqual(priority_rows[1]["priority_bucket"], "near_favorite_zone")
        self.assertIn("tracker-only", priority_rows[1]["favorite_score_gate_note"])
        self.assertEqual(feature_rows_by_symbol["NOK"]["priority_bucket"], "favorite_setup")
        self.assertEqual(feature_rows_by_symbol["CHOP"]["priority_bucket"], "near_favorite_zone")

    def test_post_earnings_hard_rule_blocks_non_post_earnings_setups_inside_10_sessions(self):
        priority_rows = [
            {
                "symbol": "ARIS",
                "side": "LONG",
                "score": 140,
                "setup_family": "avwap_retest_followthrough",
                "retest_followthrough": True,
                "previous_anchor_path_clear": True,
                "favorite_signals": [],
                "context_signals": [],
                "latest_release_gap_date": "2026-05-01",
                "latest_release_sessions_since_gap": 4,
            },
            {
                "symbol": "BRKR",
                "side": "LONG",
                "score": 120,
                "setup_family": "favorite_zone_watch",
                "favorite_zone": "AVWAPE to UPPER_1",
                "favorite_signals": [],
                "context_signals": [],
                "latest_release_gap_date": "2026-05-01",
                "latest_release_sessions_since_gap": 7,
            },
            {
                "symbol": "NVDA",
                "side": "LONG",
                "score": 130,
                "setup_family": "post_earnings_52w_break",
                "post_earnings_active": True,
                "post_earnings_sessions_since_gap": 4,
                "favorite_signals": [master_avwap.POST_EARNINGS_BREAK_SIGNAL],
                "context_signals": [],
                "latest_release_gap_date": "2026-05-01",
                "latest_release_sessions_since_gap": 4,
            },
            {
                "symbol": "NOK",
                "side": "LONG",
                "score": 122,
                "setup_family": master_avwap.AVWAPE_TO_FIRST_DEV_FAMILY,
                "favorite_zone": "AVWAPE to UPPER_1",
                "favorite_signals": ["CROSS_UP_UPPER_1"],
                "context_signals": [],
                "latest_release_earnings_date": "2026-04-30",
                "latest_known_earnings_date": "2026-04-30",
                "latest_known_earnings_sessions_since": 10,
                "latest_known_earnings_in_post_window": True,
                "latest_release_gap_date": "2026-05-01",
                "latest_release_sessions_since_gap": 10,
            },
        ]
        ai_state = {
            "symbols": {
                "ARIS": {"setup_family": "avwap_retest_followthrough"},
                "BRKR": {"setup_family": "favorite_zone_watch"},
                "NVDA": {"setup_family": "post_earnings_52w_break"},
                "NOK": {"setup_family": master_avwap.AVWAPE_TO_FIRST_DEV_FAMILY},
            }
        }
        feature_rows_by_symbol = {"ARIS": {}, "BRKR": {}, "NVDA": {}, "NOK": {}}

        master_avwap.apply_post_earnings_hard_rule_blocks(
            priority_rows,
            ai_state,
            feature_rows_by_symbol,
        )
        master_avwap.apply_final_priority_buckets(priority_rows, ai_state, [], feature_rows_by_symbol)

        self.assertTrue(priority_rows[0]["post_earnings_hard_rule_blocked"])
        self.assertTrue(priority_rows[0]["ranking_blocked"])
        self.assertEqual(priority_rows[0]["priority_bucket"], "")
        self.assertIn("post-earnings hard rule", priority_rows[0]["ranking_block_reason"])
        self.assertTrue(priority_rows[1]["post_earnings_hard_rule_blocked"])
        self.assertEqual(priority_rows[1]["priority_bucket"], "")
        self.assertFalse(priority_rows[2]["post_earnings_hard_rule_blocked"])
        self.assertEqual(priority_rows[2]["priority_bucket"], "near_favorite_zone")
        self.assertFalse(priority_rows[3]["post_earnings_hard_rule_blocked"])
        self.assertEqual(priority_rows[3]["priority_bucket"], "near_favorite_zone")

    def test_post_earnings_hard_rule_blocks_stale_gap_context_with_fresh_known_earnings(self):
        priority_rows = [
            {
                "symbol": "XYZ",
                "side": "LONG",
                "score": 150,
                "setup_family": "mid_earnings_ema15_retest",
                "favorite_signals": [master_avwap.MID_EARNINGS_EMA15_RETEST_SIGNAL],
                "context_signals": [],
                "latest_release_gap_date": "2026-02-27",
                "latest_release_sessions_since_gap": 43,
                "latest_known_earnings_date": "2026-05-07",
                "latest_known_earnings_calendar_days_since": 1,
            }
        ]
        ai_state = {
            "symbols": {
                "XYZ": {
                    "setup_family": "mid_earnings_ema15_retest",
                    "latest_release_gap_date": "2026-02-27",
                    "latest_release_sessions_since_gap": 43,
                    "latest_known_earnings_date": "2026-05-07",
                    "latest_known_earnings_calendar_days_since": 1,
                }
            }
        }
        feature_rows_by_symbol = {"XYZ": {}}

        master_avwap.apply_post_earnings_hard_rule_blocks(
            priority_rows,
            ai_state,
            feature_rows_by_symbol,
        )

        self.assertTrue(priority_rows[0]["post_earnings_hard_rule_blocked"])
        self.assertTrue(priority_rows[0]["ranking_blocked"])
        self.assertEqual(priority_rows[0]["priority_bucket"], "")
        self.assertIn("daily earnings-gap context was unavailable", priority_rows[0]["ranking_block_reason"])

    def test_pre_earnings_priority_block_keeps_theta_d1_exception(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "master_avwap_d1_watchlist.json"
            priority_rows = [
                {
                    "symbol": "AAPL",
                    "side": "LONG",
                    "priority_bucket": "favorite_setup",
                    "score": 120,
                    "pre_earnings_setup_blocked": True,
                    "pre_earnings_setup_block_reason": "non-theta setup blocked",
                }
            ]
            theta_rows = [
                {
                    "symbol": "AAPL",
                    "score": 88,
                    "option_status": "recommended",
                    "best_option": {
                        "status": "recommended",
                        "credit": 0.35,
                        "strike": 95.0,
                        "expiration": "20260508",
                        "market_days": 3,
                        "contracts_needed_for_100": 3,
                        "covered_support_count": 3,
                        "total_support_count": 4,
                    },
                    "support_summary": "SMA_20@100.00",
                }
            ]

            payload = update_master_avwap_d1_watchlist(path, priority_rows, theta_rows, [], {"symbols": {}})

        entry = payload["symbols"]["AAPL"]
        self.assertIn("sold_put_premium_viable", entry["watch_reasons"])
        self.assertNotIn("favorite_setup", entry["watch_reasons"])

    def test_d1_watchlist_preloads_first_dev_and_52w_trigger_levels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "master_avwap_d1_watchlist.json"
            priority_rows = [
                {
                    "symbol": "AAPL",
                    "side": "LONG",
                    "priority_bucket": "favorite_setup",
                    "score": 240,
                    "favorite_zone": "AVWAPE to UPPER_1",
                    "setup_family": "favorite_zone_watch",
                },
                {
                    "symbol": "TSLA",
                    "side": "SHORT",
                    "priority_bucket": "near_favorite_zone",
                    "score": 210,
                    "favorite_zone": "LOWER_1 to AVWAPE",
                    "setup_family": "favorite_zone_watch",
                },
                {
                    "symbol": "NVDA",
                    "side": "LONG",
                    "priority_bucket": "near_favorite_zone",
                    "score": 225,
                    "post_earnings_active": True,
                    "post_earnings_monitor_level": 120.0,
                    "post_earnings_monitor_level_label": "52W_HIGH",
                    "post_earnings_note": "new 52-week high 120.00; waiting for fresh break",
                    "setup_family": "post_earnings_52w_break",
                },
            ]
            ai_state = {
                "symbols": {
                    "AAPL": {
                        "last_close": 101.0,
                        "current_anchor": {"date": "2026-04-01", "vwap": 100.0, "bands": {"UPPER_1": 102.0}},
                    },
                    "TSLA": {
                        "last_close": 99.0,
                        "current_anchor": {"date": "2026-04-02", "vwap": 100.0, "bands": {"LOWER_1": 98.0}},
                    },
                    "NVDA": {"last_close": 119.0},
                }
            }

            payload = update_master_avwap_d1_watchlist(path, priority_rows, [], [], ai_state)

        aapl_trigger = payload["symbols"]["AAPL"]["trigger_levels"][0]
        self.assertEqual(aapl_trigger["label"], "UPPER_1")
        self.assertEqual(aapl_trigger["action"], "break_above")
        self.assertEqual(aapl_trigger["level"], 102.0)
        aapl_retest_trigger = payload["symbols"]["AAPL"]["trigger_levels"][1]
        self.assertEqual(aapl_retest_trigger["event_type"], "avwape_retest_watch")
        self.assertEqual(aapl_retest_trigger["label"], "AVWAPE")
        self.assertEqual(aapl_retest_trigger["action"], "break_below")
        self.assertEqual(aapl_retest_trigger["level"], 100.0)

        tsla_trigger = payload["symbols"]["TSLA"]["trigger_levels"][0]
        self.assertEqual(tsla_trigger["label"], "LOWER_1")
        self.assertEqual(tsla_trigger["action"], "break_below")
        self.assertEqual(tsla_trigger["level"], 98.0)
        tsla_retest_trigger = payload["symbols"]["TSLA"]["trigger_levels"][1]
        self.assertEqual(tsla_retest_trigger["event_type"], "avwape_retest_watch")
        self.assertEqual(tsla_retest_trigger["label"], "AVWAPE")
        self.assertEqual(tsla_retest_trigger["action"], "break_above")
        self.assertEqual(tsla_retest_trigger["level"], 100.0)

        nvda_trigger = payload["symbols"]["NVDA"]["trigger_levels"][0]
        self.assertEqual(nvda_trigger["event_type"], "post_earnings_52w_break")
        self.assertEqual(nvda_trigger["label"], "52W_HIGH")
        self.assertEqual(nvda_trigger["level"], 120.0)

    def test_d1_watchlist_preloads_mid_earnings_retest_levels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "master_avwap_d1_watchlist.json"
            priority_rows = [
                {
                    "symbol": "AAPL",
                    "side": "LONG",
                    "priority_bucket": "near_favorite_zone",
                    "score": 225,
                    "setup_family": "mid_earnings_above_2nd_stdev",
                    "mid_earnings_active_second_stdev_hold": True,
                },
                {
                    "symbol": "TSLA",
                    "side": "SHORT",
                    "priority_bucket": "near_favorite_zone",
                    "score": 215,
                    "setup_family": "mid_earnings_above_2nd_stdev",
                    "mid_earnings_active_second_stdev_hold": True,
                },
            ]
            ai_state = {
                "symbols": {
                    "AAPL": {
                        "last_close": 125.0,
                        "entry_feature_snapshot": {"ema15": 120.0},
                        "current_anchor": {"date": "2026-04-01", "bands": {"UPPER_1": 116.0}},
                    },
                    "TSLA": {
                        "last_close": 75.0,
                        "entry_feature_snapshot": {"ema15": 80.0},
                        "current_anchor": {"date": "2026-04-01", "bands": {"LOWER_1": 84.0}},
                    },
                }
            }

            payload = update_master_avwap_d1_watchlist(path, priority_rows, [], [], ai_state)

        aapl_triggers = {item["label"]: item for item in payload["symbols"]["AAPL"]["trigger_levels"]}
        self.assertEqual(aapl_triggers["EMA_15"]["action"], "break_below")
        self.assertEqual(aapl_triggers["EMA_15"]["event_type"], "mid_earnings_ema15_retest_watch")
        self.assertEqual(aapl_triggers["UPPER_1"]["action"], "break_below")
        self.assertEqual(aapl_triggers["UPPER_1"]["level"], 116.0)

        tsla_triggers = {item["label"]: item for item in payload["symbols"]["TSLA"]["trigger_levels"]}
        self.assertEqual(tsla_triggers["EMA_15"]["action"], "break_above")
        self.assertEqual(tsla_triggers["LOWER_1"]["action"], "break_above")
        self.assertEqual(tsla_triggers["LOWER_1"]["event_type"], "mid_earnings_first_dev_retest_watch")

    def test_d1_watchlist_records_a_s_upgrade_targets_for_scan_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "master_avwap_d1_watchlist.json"
            priority_rows = [
                {
                    "symbol": "MSFT",
                    "side": "LONG",
                    "priority_bucket": "",
                    "score": 72,
                    "setup_family": "general_watch",
                    "previous_day_high": 105.0,
                    "trendline_candidate": {"current_line_price": 106.0, "end_date": "2026-04-02"},
                    "sma_obstacles": [
                        {"label": "SMA_50", "level": 107.0},
                        {"label": "SMA_200", "level": 108.0},
                    ],
                    "previous_anchor_obstacles": [
                        {"label": "PREV_UPPER_1", "level": 109.0},
                        {"label": "PREV_UPPER_2", "level": 110.0},
                    ],
                }
            ]
            ai_state = {
                "symbols": {
                    "MSFT": {
                        "side": "LONG",
                        "last_close": 100.0,
                        "current_anchor": {
                            "date": "2026-04-01",
                            "vwap": 98.0,
                            "bands": {"UPPER_1": 103.0, "UPPER_2": 108.0},
                        },
                        "entry_feature_snapshot": {"ema15": 96.0},
                        "previous_day_high": 105.0,
                    }
                }
            }

            payload = update_master_avwap_d1_watchlist(path, priority_rows, [], [], ai_state)

        entry = payload["symbols"]["MSFT"]
        self.assertIn("a_s_upgrade_target", entry["watch_reasons"])
        self.assertTrue(entry["upgrade_targets"])
        self.assertIn("1st-dev break", entry["upgrade_summary"])
        labels = {target["label"] for target in entry["upgrade_targets"]}
        self.assertIn("UPPER_1", labels)
        self.assertGreater(len(entry["upgrade_targets"]), 4)
        self.assertEqual(len(entry["trigger_levels"]), len(entry["upgrade_targets"]))
        self.assertIn("+", entry["upgrade_summary"])
        self.assertTrue(all(target.get("target_tier") == "A/S" for target in entry["upgrade_targets"]))

    def test_d1_watchlist_drops_stale_post_earnings_avwape_bounce(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "master_avwap_d1_watchlist.json"
            stale_date = (datetime.now().date() - timedelta(days=3)).isoformat()
            master_avwap.save_json(
                path,
                {
                    "symbols": {
                        "AAPL": {
                            "symbol": "AAPL",
                            "side": "LONG",
                            "first_seen": stale_date,
                            "last_seen": stale_date,
                            "active_current_scan": False,
                            "setup_family": "post_earnings_avwap_bounce",
                            "post_earnings_bounce_date": stale_date,
                            "post_earnings_bounce_age_sessions": 3,
                            "watch_reasons": ["post_earnings_active"],
                        }
                    }
                },
                pretty=True,
            )

            payload = master_avwap.update_master_avwap_d1_watchlist(path, [], [], [], {"symbols": {}})

        self.assertNotIn("AAPL", payload["symbols"])

    def test_master_scan_symbols_include_d1_watchlist_by_side(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "master_avwap_d1_watchlist.json"
            reference_date = date(2026, 5, 10)
            master_avwap.save_json(
                path,
                {
                    "symbols": {
                        "AAPL": {"symbol": "AAPL", "side": "SHORT", "last_seen": reference_date.isoformat()},
                        "MSFT": {"symbol": "MSFT", "side": "LONG", "last_seen": reference_date.isoformat()},
                        "TSLA": {"symbol": "TSLA", "side": "SHORT", "last_seen": reference_date.isoformat()},
                        "OLD": {"symbol": "OLD", "side": "LONG", "last_seen": "2026-04-20"},
                    }
                },
                pretty=True,
            )

            longs, shorts, added = master_avwap.append_master_avwap_d1_watchlist_symbols(
                ["AAPL"],
                ["NVDA"],
                watchlist_path=path,
                reference_date=reference_date,
            )

        self.assertEqual(added, 2)
        self.assertEqual(longs, ["AAPL", "MSFT"])
        self.assertEqual(shorts, ["NVDA", "TSLA"])

    def test_save_json_serializes_date_like_runtime_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "payload.json"

            master_avwap.save_json(
                path,
                {
                    "run_date": date(2026, 5, 12),
                    "generated_at": datetime(2026, 5, 12, 17, 55, 44),
                    "pandas_timestamp": pd.Timestamp("2026-05-12 17:55:44"),
                    "path": Path("runtime/ai_state.json"),
                    "symbols": {"AAPL", "MSFT"},
                },
                pretty=True,
            )

            payload = master_avwap.load_json(path, default={})

        self.assertEqual(payload["run_date"], "2026-05-12")
        self.assertEqual(payload["generated_at"], "2026-05-12T17:55:44")
        self.assertEqual(payload["pandas_timestamp"], "2026-05-12T17:55:44")
        self.assertEqual(payload["path"], "runtime\\ai_state.json" if sys.platform.startswith("win") else "runtime/ai_state.json")
        self.assertEqual(payload["symbols"], ["AAPL", "MSFT"])

    def test_ib_status_and_request_failures_are_not_logged_as_errors(self):
        ib = master_avwap.IBApi()
        ib.ready[123] = False

        with (
            patch.object(master_avwap.logging, "error") as log_error,
            patch.object(master_avwap.logging, "info") as log_info,
            patch.object(master_avwap.logging, "warning") as log_warning,
        ):
            ib.error(-1, 2108, "Market data farm connection is inactive but should be available upon demand.usopt")
            ib.error(123, 200, "No security definition has been found for the request")

        log_error.assert_not_called()
        log_info.assert_called_once()
        log_warning.assert_called_once()
        self.assertTrue(ib.ready[123])

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

    def test_scan_factor_observations_compute_side_and_spy_relative_returns(self):
        history = pd.DataFrame(
            [
                {
                    "run_id": "run-1",
                    "run_timestamp": "2026-01-02T16:00:00",
                    "run_date": "2026-01-02",
                    "watchlist_label": "test",
                    "symbol": "AAPL",
                    "side": "LONG",
                    "last_trade_date": "2026-01-02",
                    "last_close": 100.0,
                },
                {
                    "run_id": "run-2",
                    "run_timestamp": "2026-01-05T16:00:00",
                    "run_date": "2026-01-05",
                    "watchlist_label": "test",
                    "symbol": "AAPL",
                    "side": "LONG",
                    "last_trade_date": "2026-01-05",
                    "last_close": 110.0,
                },
                {
                    "run_id": "run-1",
                    "run_timestamp": "2026-01-02T16:00:00",
                    "run_date": "2026-01-02",
                    "watchlist_label": "test",
                    "symbol": "TSLA",
                    "side": "SHORT",
                    "last_trade_date": "2026-01-02",
                    "last_close": 100.0,
                },
                {
                    "run_id": "run-2",
                    "run_timestamp": "2026-01-05T16:00:00",
                    "run_date": "2026-01-05",
                    "watchlist_label": "test",
                    "symbol": "TSLA",
                    "side": "SHORT",
                    "last_trade_date": "2026-01-05",
                    "last_close": 90.0,
                },
                {
                    "run_id": "run-1",
                    "run_timestamp": "2026-01-02T16:00:00",
                    "run_date": "2026-01-02",
                    "watchlist_label": "test",
                    "symbol": "SPY",
                    "side": "LONG",
                    "last_trade_date": "2026-01-02",
                    "last_close": 100.0,
                },
                {
                    "run_id": "run-2",
                    "run_timestamp": "2026-01-05T16:00:00",
                    "run_date": "2026-01-05",
                    "watchlist_label": "test",
                    "symbol": "SPY",
                    "side": "LONG",
                    "last_trade_date": "2026-01-05",
                    "last_close": 105.0,
                },
            ]
        )

        rows = build_scan_factor_observation_rows(history, horizons=(1,))
        by_symbol = {row["symbol"]: row for row in rows if row["scan_date"] == "2026-01-02"}

        self.assertAlmostEqual(by_symbol["AAPL"]["raw_return_pct"], 10.0)
        self.assertAlmostEqual(by_symbol["AAPL"]["side_return_pct"], 10.0)
        self.assertAlmostEqual(by_symbol["AAPL"]["spy_relative_side_return_pct"], 5.0)
        self.assertAlmostEqual(by_symbol["TSLA"]["raw_return_pct"], -10.0)
        self.assertAlmostEqual(by_symbol["TSLA"]["side_return_pct"], 10.0)
        self.assertAlmostEqual(by_symbol["TSLA"]["spy_relative_side_return_pct"], 15.0)

    def test_scan_factor_leaderboard_ranks_recent_working_attributes(self):
        rows = []
        for idx in range(8):
            symbol = f"WIN{idx}"
            rows.extend(
                [
                    {
                        "run_id": "run-1",
                        "run_timestamp": "2026-01-02T16:00:00",
                        "run_date": "2026-01-02",
                        "watchlist_label": "test",
                        "symbol": symbol,
                        "side": "LONG",
                        "last_trade_date": "2026-01-02",
                        "last_close": 100.0,
                        "current_band_zone": "CLEAN",
                        "priority_score": 130.0,
                        "events_today": "BOUNCE_VWAP;CROSS_UP_VWAP",
                    },
                    {
                        "run_id": "run-2",
                        "run_timestamp": "2026-01-05T16:00:00",
                        "run_date": "2026-01-05",
                        "watchlist_label": "test",
                        "symbol": symbol,
                        "side": "LONG",
                        "last_trade_date": "2026-01-05",
                        "last_close": 110.0,
                        "current_band_zone": "CLEAN",
                        "priority_score": 130.0,
                    },
                ]
            )
        for idx in range(8):
            symbol = f"BAD{idx}"
            rows.extend(
                [
                    {
                        "run_id": "run-1",
                        "run_timestamp": "2026-01-02T16:00:00",
                        "run_date": "2026-01-02",
                        "watchlist_label": "test",
                        "symbol": symbol,
                        "side": "LONG",
                        "last_trade_date": "2026-01-02",
                        "last_close": 100.0,
                        "current_band_zone": "MESSY",
                        "priority_score": 70.0,
                    },
                    {
                        "run_id": "run-2",
                        "run_timestamp": "2026-01-05T16:00:00",
                        "run_date": "2026-01-05",
                        "watchlist_label": "test",
                        "symbol": symbol,
                        "side": "LONG",
                        "last_trade_date": "2026-01-05",
                        "last_close": 95.0,
                        "current_band_zone": "MESSY",
                        "priority_score": 70.0,
                    },
                ]
            )
        history = pd.DataFrame(rows)
        observations = build_scan_factor_observation_rows(history, horizons=(1,))
        leaderboard = build_scan_factor_leaderboard_rows(
            history,
            observations,
            min_observations=8,
            reference_date="2026-01-02",
        )

        clean_row = next(
            row
            for row in leaderboard
            if row["factor_key"] == "current_band_zone" and row["value_label"] == "CLEAN"
        )
        messy_row = next(
            row
            for row in leaderboard
            if row["factor_key"] == "current_band_zone" and row["value_label"] == "MESSY"
        )
        event_row = next(
            row
            for row in leaderboard
            if row["factor_key"] == "events_today" and row["value_label"] == "BOUNCE_VWAP"
        )
        score_bucket_row = next(
            row
            for row in leaderboard
            if row["factor_key"] == "priority_score" and row["value_label"] == "120 to < 140"
        )

        self.assertEqual(clean_row["observation_count"], 8)
        self.assertGreater(clean_row["side_return_edge_pct"], 0)
        self.assertLess(messy_row["side_return_edge_pct"], 0)
        self.assertEqual(event_row["observation_count"], 8)
        self.assertEqual(score_bucket_row["observation_count"], 8)

    def test_export_scan_factor_views_writes_observation_and_leaderboard_csvs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            history_path = temp_path / "d1_features_history.csv"
            observations_path = temp_path / "scan_factor_observations.csv"
            leaderboard_path = temp_path / "scan_factor_leaderboard.csv"
            pd.DataFrame(
                [
                    {
                        "run_id": "run-1",
                        "run_timestamp": "2026-01-02T16:00:00",
                        "run_date": "2026-01-02",
                        "watchlist_label": "test",
                        "symbol": "AAPL",
                        "side": "LONG",
                        "last_trade_date": "2026-01-02",
                        "last_close": 100.0,
                        "current_band_zone": "CLEAN",
                    },
                    {
                        "run_id": "run-2",
                        "run_timestamp": "2026-01-05T16:00:00",
                        "run_date": "2026-01-05",
                        "watchlist_label": "test",
                        "symbol": "AAPL",
                        "side": "LONG",
                        "last_trade_date": "2026-01-05",
                        "last_close": 103.0,
                        "current_band_zone": "CLEAN",
                    },
                ]
            ).to_csv(history_path, index=False)

            result = export_scan_factor_views(
                history_path=history_path,
                observations_path=observations_path,
                leaderboard_path=leaderboard_path,
                min_observations=1,
            )

            self.assertEqual(result["observation_count"], 1)
            self.assertTrue(observations_path.exists())
            self.assertTrue(leaderboard_path.exists())
            observations = pd.read_csv(observations_path)
            leaderboard = pd.read_csv(leaderboard_path)
            self.assertIn("side_return_pct", observations.columns)
            self.assertIn("factor_key", leaderboard.columns)
            self.assertEqual(leaderboard.loc[0, "factor_key"], "current_band_zone")

    def test_bot_tier_tracker_builds_s_a_outcomes_and_catch_rate(self):
        rows = []

        def add_symbol(symbol, bucket, zone, start_close, end_close, score):
            rows.extend(
                [
                    {
                        "run_id": "run-1",
                        "run_timestamp": "2026-01-02T16:00:00",
                        "run_date": "2026-01-02",
                        "watchlist_label": "test",
                        "symbol": symbol,
                        "side": "LONG",
                        "last_trade_date": "2026-01-02",
                        "last_close": start_close,
                        "priority_bucket": bucket,
                        "priority_score": score,
                        "setup_family": "avwap_breakout" if bucket else "general_watch",
                        "current_band_zone": zone,
                    },
                    {
                        "run_id": "run-2",
                        "run_timestamp": "2026-01-05T16:00:00",
                        "run_date": "2026-01-05",
                        "watchlist_label": "test",
                        "symbol": symbol,
                        "side": "LONG",
                        "last_trade_date": "2026-01-05",
                        "last_close": end_close,
                        "priority_bucket": bucket,
                        "priority_score": score,
                        "setup_family": "avwap_breakout" if bucket else "general_watch",
                        "current_band_zone": zone,
                    },
                ]
            )

        for idx in range(4):
            add_symbol(f"S{idx}", "favorite_setup", "CLEAN", 100.0, 112.0, 140.0)
        for idx in range(2):
            add_symbol(f"A{idx}", "near_favorite_zone", "CLEAN", 100.0, 108.0, 115.0)
        for idx in range(4):
            add_symbol(f"MISS{idx}", "", "CLEAN", 100.0, 112.0, 70.0)
        for idx in range(6):
            add_symbol(f"BAD{idx}", "", "MESSY", 100.0, 94.0, 70.0)

        history = pd.DataFrame(rows)
        observations = build_scan_factor_observation_rows(history, horizons=(1,))
        leaderboard = build_scan_factor_leaderboard_rows(
            history,
            observations,
            min_observations=4,
            reference_date="2026-01-02",
        )

        tier_picks = build_bot_tier_pick_rows(history, leaderboard)
        tier_outcomes = build_bot_tier_outcome_rows(history, observations, leaderboard)
        performance = build_bot_tier_performance_rows(
            tier_outcomes,
            observations,
            reference_date="2026-01-02",
        )
        catch_rate = build_bot_tier_catch_rate_rows(
            history,
            observations,
            leaderboard,
            reference_date="2026-01-02",
        )

        tiers = {row["symbol"]: row["tier"] for row in tier_picks}
        self.assertEqual(tiers["S0"], "S")
        self.assertEqual(tiers["A0"], "A")
        self.assertTrue(all(row["positive_scan_factor_match_count"] > 0 for row in tier_outcomes))
        summary = next(
            row
            for row in performance
            if row["tier"] == "S/A" and row["side"] == "ALL" and row["horizon_sessions"] == 1
        )
        self.assertEqual(summary["observation_count"], 6)
        self.assertGreater(summary["avg_side_return_pct"], 0)
        catch_summary = next(
            row
            for row in catch_rate
            if row["side"] == "ALL" and row["horizon_sessions"] == 1
        )
        self.assertEqual(catch_summary["factor_winner_count"], 10)
        self.assertEqual(catch_summary["caught_winner_count"], 6)
        self.assertEqual(catch_summary["missed_winner_count"], 4)

    def test_export_bot_tier_tracker_views_writes_tier_csvs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            history_path = temp_path / "d1_features_history.csv"
            tier_list_path = temp_path / "tier_list.csv"
            tier_outcomes_path = temp_path / "tier_outcomes.csv"
            tier_performance_path = temp_path / "tier_performance.csv"
            tier_catch_rate_path = temp_path / "tier_catch_rate.csv"
            pd.DataFrame(
                [
                    {
                        "run_id": "run-1",
                        "run_timestamp": "2026-01-02T16:00:00",
                        "run_date": "2026-01-02",
                        "watchlist_label": "test",
                        "symbol": "AAPL",
                        "side": "LONG",
                        "last_trade_date": "2026-01-02",
                        "last_close": 100.0,
                        "priority_bucket": "favorite_setup",
                        "priority_score": 150.0,
                        "current_band_zone": "CLEAN",
                    },
                    {
                        "run_id": "run-2",
                        "run_timestamp": "2026-01-05T16:00:00",
                        "run_date": "2026-01-05",
                        "watchlist_label": "test",
                        "symbol": "AAPL",
                        "side": "LONG",
                        "last_trade_date": "2026-01-05",
                        "last_close": 104.0,
                        "priority_bucket": "favorite_setup",
                        "priority_score": 150.0,
                        "current_band_zone": "CLEAN",
                    },
                ]
            ).to_csv(history_path, index=False)

            result = export_bot_tier_tracker_views(
                history_path=history_path,
                tier_list_path=tier_list_path,
                tier_outcomes_path=tier_outcomes_path,
                tier_performance_path=tier_performance_path,
                tier_catch_rate_path=tier_catch_rate_path,
                min_observations=1,
            )

            self.assertEqual(result["tier_pick_count"], 1)
            self.assertEqual(result["tier_outcome_count"], 1)
            self.assertTrue(tier_list_path.exists())
            self.assertTrue(tier_outcomes_path.exists())
            self.assertTrue(tier_performance_path.exists())
            self.assertTrue(tier_catch_rate_path.exists())
            tier_list = pd.read_csv(tier_list_path)
            tier_outcomes = pd.read_csv(tier_outcomes_path)
            self.assertEqual(tier_list.loc[0, "tier"], "S")
            self.assertIn("side_return_pct", tier_outcomes.columns)

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

    def test_theta_put_candidate_requires_support_stack_and_earnings_buffer(self):
        dates = pd.bdate_range("2025-08-01", periods=180)
        rows = []
        for idx, dt_value in enumerate(dates):
            close_price = 60.0 + (idx * 0.22)
            rows.append(
                {
                    "datetime": dt_value,
                    "open": close_price - 0.2,
                    "high": close_price + 0.8,
                    "low": close_price - 0.8,
                    "close": close_price,
                    "volume": 1_000_000,
                }
            )
        df = pd.DataFrame(rows)
        indicator_row = compute_indicator_frame(df).iloc[-1]
        last_trade_date = df.iloc[-1]["datetime"].date()
        last_close = float(df.iloc[-1]["close"])

        candidate = evaluate_theta_put_candidate(
            symbol="MU",
            side="LONG",
            df=df,
            last_trade_date=last_trade_date,
            last_close=last_close,
            atr20=10.0,
            current_anchor_meta={"vwap": last_close - 3.0, "bands": {"LOWER_1": last_close - 6.0, "UPPER_1": last_close - 1.5}},
            previous_anchor_meta={"vwap": last_close - 4.0, "bands": {"LOWER_1": last_close - 8.0, "UPPER_1": last_close - 2.5}},
            indicator_row=indicator_row,
            compression_summary={"is_compressed": False},
            recent_earnings_dates=[(last_trade_date - timedelta(days=35)).isoformat()],
            upcoming_earnings_dates=[(last_trade_date + timedelta(days=35)).isoformat()],
        )

        self.assertIsNotNone(candidate)
        self.assertGreaterEqual(candidate["support_count"], 3)
        self.assertGreaterEqual(candidate["major_sma_support_count"], 1)
        self.assertGreaterEqual(candidate["avwap_support_count"], 1)
        self.assertGreaterEqual(candidate["previous_first_dev_support_count"], 1)
        self.assertEqual(candidate["side"], "LONG")
        self.assertIn("SMA_50", candidate["major_sma_supports"])
        self.assertTrue(candidate["top_score_drivers"])
        self.assertIn("major SMA support", candidate["top_score_drivers"])
        self.assertIn("previous 1st-dev support", candidate["top_score_drivers"])
        self.assertIsInstance(candidate["risk_flags"], list)
        self.assertNotIn("SMA_20", [support["label"] for support in candidate["supports"]])

    def test_theta_candidates_require_major_sma_support(self):
        dates = pd.bdate_range("2025-08-01", periods=180)
        rows = []
        for idx, dt_value in enumerate(dates):
            close_price = 60.0 + (idx * 0.22)
            rows.append(
                {
                    "datetime": dt_value,
                    "open": close_price - 0.2,
                    "high": close_price + 0.8,
                    "low": close_price - 0.8,
                    "close": close_price,
                    "volume": 1_000_000,
                }
            )
        df = pd.DataFrame(rows)
        last_trade_date = df.iloc[-1]["datetime"].date()
        last_close = float(df.iloc[-1]["close"])
        indicator_row = {"sma_50": None, "sma_100": None, "sma_200": None}
        common_kwargs = {
            "symbol": "MU",
            "side": "LONG",
            "df": df,
            "last_trade_date": last_trade_date,
            "last_close": last_close,
            "atr20": 10.0,
            "current_anchor_meta": {"vwap": last_close - 3.0, "bands": {"LOWER_1": last_close - 6.0, "UPPER_1": last_close - 1.5}},
            "previous_anchor_meta": {"vwap": last_close - 4.0, "bands": {"LOWER_1": last_close - 8.0, "UPPER_1": last_close - 2.5}},
            "indicator_row": indicator_row,
            "compression_summary": {"is_compressed": False},
            "recent_earnings_dates": [(last_trade_date - timedelta(days=35)).isoformat()],
            "upcoming_earnings_dates": [(last_trade_date + timedelta(days=35)).isoformat()],
        }

        self.assertIsNone(evaluate_theta_put_candidate(**common_kwargs))
        self.assertIsNone(evaluate_theta_pcs_candidate(**common_kwargs))

    def test_theta_put_candidate_requires_major_sma_support(self):
        dates = pd.bdate_range("2025-08-01", periods=80)
        rows = []
        for idx, dt_value in enumerate(dates):
            close_price = 100.0 + (idx * 0.01)
            rows.append(
                {
                    "datetime": dt_value,
                    "open": close_price - 0.2,
                    "high": close_price + 0.8,
                    "low": close_price - 0.8,
                    "close": close_price,
                    "volume": 1_000_000,
                }
            )
        df = pd.DataFrame(rows)
        last_trade_date = df.iloc[-1]["datetime"].date()
        last_close = float(df.iloc[-1]["close"])

        candidate = evaluate_theta_put_candidate(
            symbol="EXT",
            side="LONG",
            df=df,
            last_trade_date=last_trade_date,
            last_close=last_close,
            atr20=10.0,
            current_anchor_meta={
                "vwap": last_close - 1.0,
                "bands": {"LOWER_1": last_close - 2.0, "UPPER_1": last_close - 0.5},
            },
            previous_anchor_meta={
                "vwap": last_close - 3.0,
                "bands": {"LOWER_1": last_close - 4.0, "UPPER_1": last_close - 1.5},
            },
            indicator_row={
                "sma_20": last_close - 0.75,
                "sma_50": last_close - 45.0,
                "sma_100": last_close - 50.0,
                "sma_200": last_close - 55.0,
            },
            compression_summary={"is_compressed": False},
            recent_earnings_dates=[(last_trade_date - timedelta(days=35)).isoformat()],
            upcoming_earnings_dates=[(last_trade_date + timedelta(days=35)).isoformat()],
        )

        self.assertIsNone(candidate)

    def test_theta_put_candidate_rejects_recent_earnings(self):
        dates = pd.bdate_range("2025-08-01", periods=180)
        rows = []
        for idx, dt_value in enumerate(dates):
            close_price = 60.0 + (idx * 0.22)
            rows.append(
                {
                    "datetime": dt_value,
                    "open": close_price - 0.2,
                    "high": close_price + 0.8,
                    "low": close_price - 0.8,
                    "close": close_price,
                    "volume": 1_000_000,
                }
            )
        df = pd.DataFrame(rows)
        indicator_row = compute_indicator_frame(df).iloc[-1]
        last_trade_date = df.iloc[-1]["datetime"].date()
        last_close = float(df.iloc[-1]["close"])

        candidate = evaluate_theta_put_candidate(
            symbol="MU",
            side="LONG",
            df=df,
            last_trade_date=last_trade_date,
            last_close=last_close,
            atr20=10.0,
            current_anchor_meta={"vwap": last_close - 3.0, "bands": {"LOWER_1": last_close - 6.0, "UPPER_1": last_close - 1.5}},
            previous_anchor_meta={"vwap": last_close - 4.0, "bands": {"LOWER_1": last_close - 8.0, "UPPER_1": last_close - 2.5}},
            indicator_row=indicator_row,
            compression_summary={"is_compressed": False},
            recent_earnings_dates=[(last_trade_date - timedelta(days=10)).isoformat()],
            upcoming_earnings_dates=[(last_trade_date + timedelta(days=35)).isoformat()],
        )

        self.assertIsNone(candidate)

    def test_sold_put_option_ranking_requires_three_covered_supports(self):
        row = {
            "symbol": "CIEN",
            "last_close": 105.0,
            "score": 80,
            "base_score": 80,
            "supports": [
                {"label": "SMA_20", "level": 100.0, "source": "sma", "distance_atr": 0.2},
                {"label": "CURRENT_AVWAPE", "level": 98.0, "source": "avwape", "distance_atr": 0.4},
                {"label": "TRENDLINE_SUPPORT", "level": 96.0, "source": "trendline", "distance_atr": 0.7},
                {"label": "SMA_50", "level": 94.0, "source": "sma", "distance_atr": 1.0},
                {"label": "PREV_AVWAPE", "level": 92.0, "source": "previous_avwape", "distance_atr": 1.3},
            ],
        }

        candidates = master_avwap._sold_put_candidate_strikes(row, [100, 98, 96, 94, 92])
        self.assertNotIn(96.0, [candidate["strike"] for candidate in candidates])
        self.assertNotIn(98.0, [candidate["strike"] for candidate in candidates])

        credits = {94.0: 0.30, 92.0: 0.12}
        quote_rows = []
        for candidate in candidates:
            credit = credits[candidate["strike"]]
            quote_rows.append(
                {
                    **candidate,
                    "expiration": "20260508",
                    "expiration_date": date(2026, 5, 8),
                    "market_days": 4,
                    "quote": {"bid": credit - 0.01, "ask": credit + 0.01},
                }
            )

        ranked = master_avwap._rank_sold_put_option_recommendations(row, quote_rows)
        self.assertEqual(ranked[0]["strike"], 94.0)
        self.assertEqual(ranked[0]["status"], "recommended")
        self.assertEqual(ranked[0]["contracts_needed_for_100"], 4)
        self.assertEqual(ranked[0]["covered_support_count"], 3)
        self.assertEqual(ranked[0]["covered_major_sma_support_count"], 1)
        self.assertNotIn("SMA_20", ranked[0]["covered_support_summary"])

    def test_theta_option_strikes_require_covered_major_sma_support(self):
        row = {
            "symbol": "CIEN",
            "last_close": 105.0,
            "score": 80,
            "base_score": 80,
            "supports": [
                {"label": "CURRENT_AVWAPE", "level": 100.0, "source": "avwape", "distance_atr": 0.4},
                {"label": "TRENDLINE_SUPPORT", "level": 98.0, "source": "trendline", "distance_atr": 0.7},
                {"label": "PREV_AVWAPE", "level": 96.0, "source": "previous_avwape", "distance_atr": 1.0},
            ],
        }

        self.assertEqual(master_avwap._sold_put_candidate_strikes(row, [100, 98, 96, 94]), [])
        self.assertEqual(master_avwap._pcs_short_strike_candidates(row, [100, 98, 96, 94]), [])

    def test_theta_option_strikes_require_covered_avwap_family_support(self):
        row = {
            "symbol": "CIEN",
            "last_close": 105.0,
            "score": 80,
            "base_score": 80,
            "supports": [
                {"label": "SMA_50", "level": 100.0, "source": "sma", "distance_atr": 0.4},
                {"label": "SMA_100", "level": 98.0, "source": "sma", "distance_atr": 0.7},
                {"label": "TRENDLINE_SUPPORT", "level": 96.0, "source": "trendline", "distance_atr": 1.0},
                {"label": "COMPRESSION_LOW", "level": 94.0, "source": "compression", "distance_atr": 1.3},
            ],
        }

        self.assertEqual(master_avwap._sold_put_candidate_strikes(row, [100, 98, 96, 95, 94]), [])
        self.assertEqual(master_avwap._pcs_short_strike_candidates(row, [100, 98, 96, 95, 94]), [])

        row["supports"].append(
            {"label": "PREV_UPPER_1", "level": 95.0, "source": "previous_avwape", "distance_atr": 1.1}
        )
        sold_put_candidates = master_avwap._sold_put_candidate_strikes(row, [100, 98, 96, 95, 94])
        pcs_candidates = master_avwap._pcs_short_strike_candidates(row, [100, 98, 96, 95, 94])

        self.assertIn(95.0, [candidate["strike"] for candidate in sold_put_candidates])
        self.assertIn(95.0, [candidate["short_strike"] for candidate in pcs_candidates])
        self.assertEqual(sold_put_candidates[0]["covered_avwap_support_count"], 1)
        self.assertEqual(sold_put_candidates[0]["covered_previous_first_dev_support_count"], 1)

    def test_pcs_ranking_uses_two_supports_and_credit_width_target(self):
        row = {
            "symbol": "NVDA",
            "last_close": 205.0,
            "score": 75,
            "base_score": 75,
            "supports": [
                {"label": "SMA_20", "level": 200.0, "source": "sma", "distance_atr": 0.4},
                {"label": "SMA_50", "level": 195.0, "source": "sma", "distance_atr": 0.7},
                {"label": "CURRENT_AVWAPE", "level": 190.0, "source": "avwape", "distance_atr": 1.0},
            ],
        }

        short_candidates = master_avwap._pcs_short_strike_candidates(row, [195, 190, 185])
        self.assertEqual(short_candidates[0]["short_strike"], 190.0)
        self.assertNotIn(195.0, [candidate["short_strike"] for candidate in short_candidates])
        self.assertNotIn("SMA_20", short_candidates[0]["covered_support_summary"])

        ranked = master_avwap._rank_pcs_option_recommendations(
            row,
            [
                {
                    **short_candidates[0],
                    "expiration": "20260522",
                    "expiration_date": date(2026, 5, 22),
                    "market_days": 14,
                    "long_strike": 185.0,
                    "short_quote": {"bid": 2.00, "ask": 2.10},
                    "long_quote": {"bid": 0.95, "ask": 1.00},
                }
            ],
        )

        self.assertEqual(ranked[0]["status"], "recommended")
        self.assertEqual(ranked[0]["short_strike"], 190.0)
        self.assertEqual(ranked[0]["long_strike"], 185.0)
        self.assertEqual(ranked[0]["credit"], 1.0)
        self.assertEqual(ranked[0]["credit_width_pct"], 20.0)
        self.assertEqual(ranked[0]["covered_major_sma_support_count"], 1)
        self.assertTrue(
            master_avwap._pcs_row_meets_credit_target(
                {"option_status": ranked[0]["status"], "best_option": ranked[0]}
            )
        )

        below_target = master_avwap._rank_pcs_option_recommendations(
            row,
            [
                {
                    **short_candidates[0],
                    "expiration": "20260522",
                    "expiration_date": date(2026, 5, 22),
                    "market_days": 14,
                    "long_strike": 185.0,
                    "short_quote": {"bid": 1.95, "ask": 2.05},
                    "long_quote": {"bid": 0.95, "ask": 1.00},
                }
            ],
        )

        self.assertEqual(below_target[0]["credit"], 0.95)
        self.assertEqual(below_target[0]["credit_width_pct"], 19.0)
        self.assertFalse(
            master_avwap._pcs_row_meets_credit_target(
                {"option_status": below_target[0]["status"], "best_option": below_target[0]}
            )
        )

    def test_option_quote_credit_prefers_last_known_then_bid(self):
        last_credit, last_source = master_avwap._option_quote_credit_with_source(
            {"bid": 0.20, "ask": 0.40, "last": 0.55}
        )
        close_credit, close_source = master_avwap._option_quote_credit_with_source(
            {"bid": 0.20, "ask": 0.40, "close": 0.35}
        )
        bid_credit, bid_source = master_avwap._option_quote_credit_with_source(
            {"bid": 0.20, "ask": 0.40}
        )

        self.assertAlmostEqual(last_credit, 0.55)
        self.assertEqual(last_source, "last")
        self.assertAlmostEqual(close_credit, 0.35)
        self.assertEqual(close_source, "close")
        self.assertAlmostEqual(bid_credit, 0.20)
        self.assertEqual(bid_source, "bid")
        self.assertAlmostEqual(master_avwap._option_quote_mid({"bid": 0.20, "ask": 0.40}), 0.30)

    def test_sold_put_ranking_uses_last_known_price_over_wide_midpoint(self):
        row = {
            "symbol": "CIEN",
            "last_close": 105.0,
            "score": 80,
            "base_score": 80,
        }
        quote_row = {
            "strike": 95.0,
            "expiration": "20260508",
            "expiration_date": date(2026, 5, 8),
            "market_days": 4,
            "covered_support_count": 3,
            "covered_major_sma_support_count": 1,
            "covered_avwap_support_count": 1,
            "covered_previous_first_dev_support_count": 1,
            "total_support_count": 3,
            "support_quality_score": 3.0,
            "quote": {"bid": 0.01, "ask": 0.99, "last": 0.18},
        }

        ranked = master_avwap._rank_sold_put_option_recommendations(row, [quote_row])

        self.assertEqual(ranked[0]["credit"], 0.18)
        self.assertEqual(ranked[0]["credit_source"], "last")
        self.assertEqual(ranked[0]["status"], "cusp")

    def test_theta_option_client_reconnects_when_scan_client_is_unavailable(self):
        class FakeIb:
            def __init__(self, connected):
                self.connected = connected

            def isConnected(self):
                return self.connected

        existing = FakeIb(False)
        theta_client = FakeIb(True)

        with patch.object(master_avwap, "connect_daily_data_client", return_value=theta_client) as connect:
            client, owned = master_avwap.ensure_theta_option_data_client(existing)

        self.assertIs(client, theta_client)
        self.assertTrue(owned)
        connect.assert_called_once_with(
            client_id=master_avwap.THETA_OPTION_CLIENT_ID,
            startup_wait=master_avwap.THETA_OPTION_CONNECT_STARTUP_WAIT_SEC,
        )

    def test_ib_option_quote_falls_back_to_delayed_market_data(self):
        class FakeIb:
            def __init__(self):
                self.option_quotes = {}
                self.option_quotes_ready = {}
                self.market_data_types = []
                self.market_data_requests = []
                self.next_id = 10

            def isConnected(self):
                return True

            def next_request_id(self):
                self.next_id += 1
                return self.next_id

            def reqMarketDataType(self, market_data_type):
                self.market_data_types.append(market_data_type)
                self.market_data_type = market_data_type

            def reqMktData(self, req_id, contract, generic_tick_list, snapshot, regulatory_snapshot, options):
                self.market_data_requests.append((generic_tick_list, snapshot, regulatory_snapshot))
                if getattr(self, "market_data_type", None) == 3:
                    self.option_quotes[req_id] = {"bid": 0.20, "ask": 0.40}
                self.option_quotes_ready[req_id] = True

            def cancelMktData(self, req_id):
                pass

        fake_ib = FakeIb()
        quote = master_avwap._fetch_ib_option_quote(
            fake_ib,
            master_avwap.create_option_contract("ABC", "20260515", 95.0),
        )

        self.assertEqual(quote["market_data_type"], 3)
        self.assertAlmostEqual(quote["bid"], 0.20)
        self.assertEqual(fake_ib.market_data_types[:2], [1, 3])
        self.assertTrue(fake_ib.market_data_requests)
        self.assertTrue(all(request == ("", True, False) for request in fake_ib.market_data_requests))

    def test_ib_option_quote_returns_when_price_arrives_before_snapshot_end(self):
        class FakeIb:
            def __init__(self):
                self.option_quotes = {}
                self.option_quotes_ready = {}
                self.request_errors = {}
                self.market_data_types = []
                self.next_id = 20

            def isConnected(self):
                return True

            def next_request_id(self):
                self.next_id += 1
                return self.next_id

            def reqMarketDataType(self, market_data_type):
                self.market_data_types.append(market_data_type)

            def reqMktData(self, req_id, contract, generic_tick_list, snapshot, regulatory_snapshot, options):
                self.option_quotes[req_id] = {"last": 0.25}
                self.option_quotes_ready[req_id] = False

            def cancelMktData(self, req_id):
                pass

        fake_ib = FakeIb()
        with patch.object(master_avwap.time, "sleep") as sleep:
            quote = master_avwap._fetch_ib_option_quote(
                fake_ib,
                master_avwap.create_option_contract("ABC", "20260515", 95.0),
                timeout_sec=1.0,
            )

        self.assertEqual(quote["last"], 0.25)
        sleep.assert_called_once_with(master_avwap.THETA_OPTION_REQUEST_DELAY_SEC)

    def test_ib_option_quote_stops_fallback_after_security_definition_error(self):
        class FakeIb:
            def __init__(self):
                self.option_quotes = {}
                self.option_quotes_ready = {}
                self.request_errors = {}
                self.market_data_types = []
                self.next_id = 30

            def isConnected(self):
                return True

            def next_request_id(self):
                self.next_id += 1
                return self.next_id

            def reqMarketDataType(self, market_data_type):
                self.market_data_types.append(market_data_type)

            def reqMktData(self, req_id, contract, generic_tick_list, snapshot, regulatory_snapshot, options):
                self.request_errors[req_id] = [
                    {"code": 200, "message": "No security definition has been found for the request"}
                ]
                self.option_quotes_ready[req_id] = True

            def cancelMktData(self, req_id):
                pass

        fake_ib = FakeIb()
        quote = master_avwap._fetch_ib_option_quote(
            fake_ib,
            master_avwap.create_option_contract("ABC", "20260515", 95.0),
            timeout_sec=1.0,
        )

        self.assertEqual(fake_ib.market_data_types, [1])
        self.assertEqual(quote["ib_error_codes"], [200])

    def test_sold_put_enrichment_scans_lower_strikes_and_prefers_furthest_viable_premium(self):
        row = {
            "symbol": "ABC",
            "last_close": 105.0,
            "score": 80,
            "supports": [
                {"label": "CURRENT_AVWAPE", "level": 100.0, "source": "avwape", "distance_atr": 0.5},
                {"label": "SMA_50", "level": 98.0, "source": "sma", "distance_atr": 0.7},
                {"label": "PREV_AVWAPE", "level": 96.0, "source": "previous_avwape", "distance_atr": 0.9},
            ],
        }
        chain = {
            "tradingClass": "ABC",
            "multiplier": "100",
            "expirations": {"20260508"},
            "strikes": {100, 99, 98, 97, 96, 95, 94, 93, 92},
        }
        quotes = {
            96.0: {"bid": 0.04, "ask": 0.08},
            95.0: {"bid": 0.12, "ask": 0.18},
            94.0: {"bid": 0.27, "ask": 0.31},
            93.0: {"bid": 0.25, "ask": 0.25},
            92.0: {"bid": 0.08, "ask": 0.10},
        }
        quoted_strikes = []

        def quote_stub(_ib, _quote_cache, **kwargs):
            strike = float(kwargs["strike"])
            quoted_strikes.append(strike)
            return quotes.get(strike, {})

        with patch.object(master_avwap, "_fetch_theta_option_quote_cached", side_effect=quote_stub):
            master_avwap._enrich_sold_put_row_with_ib_options(
                object(),
                row,
                chain,
                {},
                date(2026, 5, 5),
            )

        self.assertIn(96.0, quoted_strikes)
        self.assertIn(93.0, quoted_strikes)
        self.assertEqual(row["option_status"], "recommended")
        self.assertEqual(row["best_option"]["strike"], 93.0)
        self.assertEqual(row["best_option"]["contracts_needed_for_100"], 4)
        self.assertGreaterEqual(len(row["option_recommendations"]), 2)
        self.assertNotIn(92.0, [option["strike"] for option in row["option_recommendations"]])

    def test_pcs_enrichment_scans_lower_short_strikes_for_viable_credit_width(self):
        row = {
            "symbol": "ABC",
            "last_close": 205.0,
            "score": 75,
            "supports": [
                {"label": "SMA_50", "level": 195.0, "source": "sma", "distance_atr": 0.6},
                {"label": "CURRENT_AVWAPE", "level": 190.0, "source": "avwape", "distance_atr": 0.9},
            ],
        }
        chain = {
            "tradingClass": "ABC",
            "multiplier": "100",
            "expirations": {"20260515"},
            "strikes": {195, 190, 185, 180, 175},
        }
        quotes = {
            190.0: {"bid": 1.00, "ask": 1.10},
            185.0: {"bid": 1.30, "ask": 1.40},
            180.0: {"bid": 0.20, "ask": 0.20},
            175.0: {"bid": 0.10, "ask": 0.10},
        }
        quoted_strikes = []

        def quote_stub(_ib, _quote_cache, **kwargs):
            strike = float(kwargs["strike"])
            quoted_strikes.append(strike)
            return quotes.get(strike, {})

        with patch.object(master_avwap, "_fetch_theta_option_quote_cached", side_effect=quote_stub):
            master_avwap._enrich_pcs_row_with_ib_options(
                object(),
                row,
                chain,
                {},
                date(2026, 5, 5),
            )

        self.assertIn(190.0, quoted_strikes)
        self.assertIn(185.0, quoted_strikes)
        self.assertEqual(row["option_status"], "recommended")
        self.assertEqual(row["best_option"]["short_strike"], 185.0)
        self.assertEqual(row["best_option"]["long_strike"], 180.0)
        self.assertGreaterEqual(row["best_option"]["credit_width_ratio"], 0.20)

    def test_fetch_daily_bars_uses_recent_cache_before_live_refresh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "daily_bars"
            master_avwap._DAILY_BAR_FRAME_CACHE.clear()
            master_avwap._DAILY_BAR_CACHE_TOUCHED_AT.clear()
            master_avwap._DAILY_BAR_LIVE_FAILURE_AT.clear()
            with (
                patch.object(master_avwap, "DAILY_BARS_CACHE_DIR", cache_dir),
                patch.object(master_avwap, "_fetch_live_daily_bars") as live_fetch,
            ):
                master_avwap._write_cached_daily_bar_frame("AL", _build_daily_bar_cache_frame())

                result = master_avwap.fetch_daily_bars(object(), "AL", 40)

            live_fetch.assert_not_called()
            self.assertFalse(result.empty)
            self.assertEqual(master_avwap._get_daily_bar_source(result), master_avwap.DAILY_BAR_SOURCE_CACHE)
            master_avwap._DAILY_BAR_FRAME_CACHE.clear()
            master_avwap._DAILY_BAR_CACHE_TOUCHED_AT.clear()
            master_avwap._DAILY_BAR_LIVE_FAILURE_AT.clear()

    def test_cached_daily_bars_reload_when_disk_cache_is_newer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "daily_bars"
            master_avwap._DAILY_BAR_FRAME_CACHE.clear()
            master_avwap._DAILY_BAR_CACHE_TOUCHED_AT.clear()
            master_avwap._DAILY_BAR_LIVE_FAILURE_AT.clear()
            with patch.object(master_avwap, "DAILY_BARS_CACHE_DIR", cache_dir):
                old_frame = _build_daily_bar_cache_frame()
                new_frame = old_frame.copy()
                new_frame.loc[new_frame.index[-1], "close"] = 999.0

                master_avwap._write_cached_daily_bar_frame("AL", old_frame)
                master_avwap._DAILY_BAR_CACHE_TOUCHED_AT["AL"] = datetime.now() - timedelta(minutes=5)
                new_frame.to_csv(cache_dir / "AL.csv", index=False)

                result = master_avwap._load_cached_daily_bar_frame("AL")

            self.assertEqual(float(result.iloc[-1]["close"]), 999.0)
            master_avwap._DAILY_BAR_FRAME_CACHE.clear()
            master_avwap._DAILY_BAR_CACHE_TOUCHED_AT.clear()
            master_avwap._DAILY_BAR_LIVE_FAILURE_AT.clear()

    def test_fetch_daily_bars_refreshes_recently_touched_stale_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "daily_bars"
            stale_frame = _build_daily_bar_cache_frame()
            stale_frame["datetime"] = stale_frame["datetime"] - pd.Timedelta(days=10)
            fresh_frame = _build_daily_bar_cache_frame()
            master_avwap._DAILY_BAR_FRAME_CACHE.clear()
            master_avwap._DAILY_BAR_CACHE_TOUCHED_AT.clear()
            master_avwap._DAILY_BAR_LIVE_FAILURE_AT.clear()
            with (
                patch.object(master_avwap, "DAILY_BARS_CACHE_DIR", cache_dir),
                patch.object(master_avwap, "MASTER_AVWAP_DAILY_BARS_DIR", Path(temp_dir) / "durable"),
                patch.object(master_avwap, "_fetch_live_daily_bars", return_value=fresh_frame) as live_fetch,
            ):
                master_avwap._write_cached_daily_bar_frame("AL", stale_frame)

                result = master_avwap.fetch_daily_bars(object(), "AL", 40)

            live_fetch.assert_called_once()
            self.assertFalse(result.empty)
            self.assertEqual(
                pd.to_datetime(result.iloc[-1]["datetime"]).date(),
                pd.to_datetime(fresh_frame.iloc[-1]["datetime"]).date(),
            )
            master_avwap._DAILY_BAR_FRAME_CACHE.clear()
            master_avwap._DAILY_BAR_CACHE_TOUCHED_AT.clear()
            master_avwap._DAILY_BAR_LIVE_FAILURE_AT.clear()

    def test_durable_store_seeds_cold_cache_and_only_delta_is_fetched(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "daily_bars"
            durable_dir = Path(temp_dir) / "durable"
            master_avwap._DAILY_BAR_FRAME_CACHE.clear()
            master_avwap._DAILY_BAR_CACHE_TOUCHED_AT.clear()
            master_avwap._DAILY_BAR_LIVE_FAILURE_AT.clear()
            with (
                patch.object(master_avwap, "DAILY_BARS_CACHE_DIR", cache_dir),
                patch.object(master_avwap, "MASTER_AVWAP_DAILY_BARS_DIR", durable_dir),
            ):
                # Persist durable history (covers the request window but ends a few
                # sessions ago), then simulate a fresh machine: no local L1 cache,
                # only the durable Drive store exists.
                history = _build_daily_bar_cache_frame(450)
                history["datetime"] = history["datetime"] - pd.Timedelta(days=10)
                master_avwap._persist_durable_daily_bars("AL", master_avwap._normalize_daily_bar_frame(history))
                self.assertTrue((durable_dir / "AL.parquet").exists())
                master_avwap._DAILY_BAR_FRAME_CACHE.clear()
                master_avwap._DAILY_BAR_CACHE_TOUCHED_AT.clear()

                # Cold start: local cache missing -> load seeds from durable store.
                seeded = master_avwap._load_cached_daily_bar_frame("AL")
                self.assertFalse(seeded.empty)
                self.assertTrue((cache_dir / "AL.csv").exists())  # L1 seeded for fast reuse

                # With history present, only the recent delta is requested (a small
                # window), never the full requested lookback.
                captured = {}

                def fake_live(ib, symbol, days):
                    captured["days"] = days
                    return _build_daily_bar_cache_frame()

                master_avwap._DAILY_BAR_FRAME_CACHE.clear()
                master_avwap._DAILY_BAR_CACHE_TOUCHED_AT.clear()
                with patch.object(master_avwap, "_fetch_live_daily_bars", side_effect=fake_live):
                    master_avwap.fetch_daily_bars(object(), "AL", 400)
                self.assertLess(captured["days"], 400)
            master_avwap._DAILY_BAR_FRAME_CACHE.clear()
            master_avwap._DAILY_BAR_CACHE_TOUCHED_AT.clear()
            master_avwap._DAILY_BAR_LIVE_FAILURE_AT.clear()

    def test_intraday_durable_store_seeds_cold_cache_and_fetches_only_delta(self):
        def hourly(n_days, end_offset_days=0):
            end = pd.Timestamp.now().normalize() - pd.Timedelta(days=end_offset_days)
            sessions = pd.bdate_range(end=end, periods=n_days)
            rows = []
            for day in sessions:
                for hour in range(9, 16):
                    rows.append(
                        {
                            "datetime": pd.Timestamp(day) + pd.Timedelta(hours=hour),
                            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0,
                        }
                    )
            return pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            l1_dir = Path(temp_dir) / "intraday"
            durable_dir = Path(temp_dir) / "durable"
            master_avwap._INTRADAY_BAR_FRAME_CACHE.clear()
            master_avwap._INTRADAY_BAR_CACHE_TOUCHED_AT.clear()
            with (
                patch.object(master_avwap, "INTRADAY_BARS_CACHE_DIR", l1_dir),
                patch.object(master_avwap, "MASTER_AVWAP_INTRADAY_BARS_DIR", durable_dir),
            ):
                # A prior machine populated the Drive store; its bars are now stale.
                history = master_avwap._normalize_intraday_bar_frame(hourly(190, end_offset_days=12))
                master_avwap._persist_durable_intraday_bars("NVDA", "1h", history)
                self.assertTrue((durable_dir / "NVDA__1h.parquet").exists())

                # Fresh machine: empty local cache, same Drive.
                master_avwap._INTRADAY_BAR_FRAME_CACHE.clear()
                master_avwap._INTRADAY_BAR_CACHE_TOUCHED_AT.clear()
                captured = {}

                def fake_live(ib, symbol, *, bar_size=None, duration=None):
                    captured["days"] = master_avwap._duration_to_days(duration, 180)
                    return hourly(6)

                with patch.object(master_avwap, "_fetch_live_intraday_bars", side_effect=fake_live):
                    merged = master_avwap.fetch_intraday_bars(object(), "NVDA")

                self.assertLess(captured["days"], 180)  # only the delta, not the full window
                self.assertGreaterEqual(len(merged), len(history))  # history preserved + extended
                self.assertTrue((l1_dir / "NVDA__1h.csv").exists())  # L1 seeded for fast reuse
            master_avwap._INTRADAY_BAR_FRAME_CACHE.clear()
            master_avwap._INTRADAY_BAR_CACHE_TOUCHED_AT.clear()

    def test_warm_durable_bar_stores_force_writes_unique_symbols(self):
        def daily_frame(n):
            dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
            return pd.DataFrame(
                {"datetime": dates, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0}
            )

        def hourly_frame(n):
            sessions = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
            rows = []
            for day in sessions:
                for hour in range(9, 16):
                    rows.append(
                        {"datetime": pd.Timestamp(day) + pd.Timedelta(hours=hour),
                         "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0}
                    )
            return pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            daily_dir = Path(temp_dir) / "daily"
            intraday_dir = Path(temp_dir) / "intraday"
            with (
                patch.object(master_avwap, "MASTER_AVWAP_DAILY_BARS_DIR", daily_dir),
                patch.object(master_avwap, "MASTER_AVWAP_INTRADAY_BARS_DIR", intraday_dir),
                patch.object(master_avwap, "fetch_daily_bars", side_effect=lambda ib, sym, days: daily_frame(420)),
                patch.object(master_avwap, "fetch_intraday_bars", side_effect=lambda ib, sym, bar_size=None: hourly_frame(190)),
            ):
                summary = master_avwap.warm_durable_bar_stores(["nvda", "AAPL", "nvda", ""], ib=None)

            self.assertEqual(summary["requested"], 2)
            self.assertEqual(summary["daily"], 2)
            self.assertEqual(summary["intraday"], 2)
            self.assertEqual(summary["failed"], [])
            self.assertTrue((daily_dir / "NVDA.parquet").exists())
            self.assertTrue((daily_dir / "AAPL.parquet").exists())
            self.assertTrue((intraday_dir / "NVDA__1h.parquet").exists())
            self.assertTrue((intraday_dir / "AAPL__1h.parquet").exists())

    def test_fetch_daily_bars_skips_when_live_refresh_returns_stale_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "daily_bars"
            stale_frame = _build_daily_bar_cache_frame()
            stale_frame["datetime"] = stale_frame["datetime"] - pd.Timedelta(days=10)
            master_avwap._DAILY_BAR_FRAME_CACHE.clear()
            master_avwap._DAILY_BAR_CACHE_TOUCHED_AT.clear()
            master_avwap._DAILY_BAR_LIVE_FAILURE_AT.clear()
            with (
                patch.object(master_avwap, "DAILY_BARS_CACHE_DIR", cache_dir),
                patch.object(master_avwap, "_fetch_live_daily_bars", return_value=stale_frame) as live_fetch,
            ):
                master_avwap._write_cached_daily_bar_frame("AL", stale_frame)

                result = master_avwap.fetch_daily_bars(object(), "AL", 40)

            live_fetch.assert_called_once()
            self.assertTrue(result.empty)
            self.assertEqual(master_avwap._get_daily_bar_source(result), master_avwap.DAILY_BAR_SOURCE_CACHE)
            self.assertIn("AL", master_avwap._DAILY_BAR_LIVE_FAILURE_AT)
            master_avwap._DAILY_BAR_FRAME_CACHE.clear()
            master_avwap._DAILY_BAR_CACHE_TOUCHED_AT.clear()
            master_avwap._DAILY_BAR_LIVE_FAILURE_AT.clear()

    def test_fetch_daily_bars_uses_cache_during_live_failure_cooldown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "daily_bars"
            master_avwap._DAILY_BAR_FRAME_CACHE.clear()
            master_avwap._DAILY_BAR_CACHE_TOUCHED_AT.clear()
            master_avwap._DAILY_BAR_LIVE_FAILURE_AT.clear()
            with (
                patch.object(master_avwap, "DAILY_BARS_CACHE_DIR", cache_dir),
                patch.object(master_avwap, "_fetch_live_daily_bars") as live_fetch,
            ):
                master_avwap._write_cached_daily_bar_frame("AL", _build_daily_bar_cache_frame())
                master_avwap._DAILY_BAR_CACHE_TOUCHED_AT["AL"] = datetime.now() - timedelta(minutes=90)
                master_avwap._DAILY_BAR_LIVE_FAILURE_AT["AL"] = datetime.now()

                result = master_avwap.fetch_daily_bars(object(), "AL", 40)

            live_fetch.assert_not_called()
            self.assertFalse(result.empty)
            self.assertEqual(master_avwap._get_daily_bar_source(result), master_avwap.DAILY_BAR_SOURCE_CACHE)
            master_avwap._DAILY_BAR_FRAME_CACHE.clear()
            master_avwap._DAILY_BAR_CACHE_TOUCHED_AT.clear()
            master_avwap._DAILY_BAR_LIVE_FAILURE_AT.clear()

    def test_yahoo_daily_bar_parser_handles_multiindex_columns(self):
        dates = pd.date_range("2026-01-01", periods=3)
        raw = pd.DataFrame(
            {
                ("Open", "AL"): [10.0, 10.5, 10.8],
                ("High", "AL"): [11.0, 11.3, 11.4],
                ("Low", "AL"): [9.8, 10.1, 10.4],
                ("Close", "AL"): [10.7, 10.9, 11.0],
                ("Adj Close", "AL"): [10.7, 10.9, 11.0],
                ("Volume", "AL"): [1000, 1200, 1100],
            },
            index=dates,
        )
        raw.index.name = "Date"

        with patch.object(master_avwap.yf, "download", return_value=raw):
            result = master_avwap.fetch_daily_bars_from_yahoo("AL", 30)

        self.assertEqual(list(result.columns), master_avwap.DAILY_BAR_COLUMNS)
        self.assertEqual(master_avwap._get_daily_bar_source(result), master_avwap.DAILY_BAR_SOURCE_YAHOO)
        self.assertEqual(float(result.iloc[-1]["close"]), 11.0)

    def test_ib_weekly_expiration_detection_uses_close_expiry_spacing(self):
        self.assertTrue(
            master_avwap._has_weekly_option_expirations(
                ["20260508", "20260515", "20260619"]
            )
        )
        self.assertFalse(
            master_avwap._has_weekly_option_expirations(
                ["20260515", "20260619", "20260717"]
            )
        )

    def test_ib_option_chain_selector_prefers_weekly_expirations(self):
        chain = master_avwap._select_ib_option_chain(
            [
                {
                    "exchange": "SMART",
                    "tradingClass": "ABC",
                    "multiplier": "100",
                    "expirations": {"20260515", "20260619", "20260717"},
                    "strikes": {90, 95, 100},
                },
                {
                    "exchange": "BOX",
                    "tradingClass": "ABC",
                    "multiplier": "100",
                    "expirations": {"20260508", "20260515", "20260619"},
                    "strikes": {90, 95},
                },
            ],
            "ABC",
        )

        self.assertEqual(chain["exchange"], "BOX")
        self.assertTrue(master_avwap._has_weekly_option_expirations(chain["expirations"]))

    def test_theta_ib_enrichment_filters_rows_without_weekly_expirations(self):
        class FakeIb:
            def reqMarketDataType(self, market_data_type):
                self.market_data_type = market_data_type

        sold_put_row = {"symbol": "ABC", "score": 80, "notes": "theta setup"}
        pcs_row = {"symbol": "ABC", "score": 75, "notes": "pcs setup"}
        sold_put_rows = [sold_put_row]
        pcs_rows = [pcs_row]
        monthly_only_chain = [
            {
                "exchange": "SMART",
                "tradingClass": "ABC",
                "multiplier": "100",
                "expirations": {"20260515", "20260619", "20260717"},
                "strikes": {90, 95, 100},
            }
        ]

        with (
            patch.object(master_avwap, "is_daily_data_client_connected", return_value=True),
            patch.object(master_avwap, "_fetch_ib_option_chain_definitions", return_value=monthly_only_chain),
            patch.object(master_avwap, "_enrich_sold_put_row_with_ib_options") as sold_put_enrich,
            patch.object(master_avwap, "_enrich_pcs_row_with_ib_options") as pcs_enrich,
        ):
            master_avwap.enrich_theta_rows_with_ib_option_premiums(
                FakeIb(),
                sold_put_rows,
                pcs_rows,
                date(2026, 5, 5),
            )

        sold_put_enrich.assert_not_called()
        pcs_enrich.assert_not_called()
        self.assertEqual(sold_put_rows, [])
        self.assertEqual(pcs_rows, [])
        self.assertEqual(sold_put_row["theta_filter_reason"], "no_weekly_options")
        self.assertEqual(pcs_row["theta_filter_reason"], "no_weekly_options")
        self.assertIn("filtered: no IBKR weekly expirations", sold_put_row["notes"])

    def test_theta_ib_enrichment_keeps_rows_with_weekly_expirations(self):
        class FakeIb:
            def reqMarketDataType(self, market_data_type):
                self.market_data_type = market_data_type

        sold_put_rows = [{"symbol": "ABC", "score": 80}]
        pcs_rows = [{"symbol": "ABC", "score": 75}]
        weekly_chain = [
            {
                "exchange": "SMART",
                "tradingClass": "ABC",
                "multiplier": "100",
                "expirations": {"20260508", "20260515", "20260619"},
                "strikes": {90, 95, 100},
            }
        ]

        def mark_sold_put(_ib, row, _chain, _quote_cache, _reference_date):
            row["ib_weekly_checked"] = "sold_put"

        def mark_pcs(_ib, row, _chain, _quote_cache, _reference_date):
            row["ib_weekly_checked"] = "pcs"
            row["option_status"] = "recommended"
            row["best_option"] = {
                "play_type": "pcs",
                "status": "recommended",
                "short_strike": 190.0,
                "long_strike": 185.0,
                "width": 5.0,
                "credit": 1.0,
                "credit_width_ratio": 0.20,
                "credit_width_pct": 20.0,
            }

        with (
            patch.object(master_avwap, "is_daily_data_client_connected", return_value=True),
            patch.object(master_avwap, "_fetch_ib_option_chain_definitions", return_value=weekly_chain),
            patch.object(master_avwap, "_enrich_sold_put_row_with_ib_options", side_effect=mark_sold_put),
            patch.object(master_avwap, "_enrich_pcs_row_with_ib_options", side_effect=mark_pcs),
        ):
            master_avwap.enrich_theta_rows_with_ib_option_premiums(
                FakeIb(),
                sold_put_rows,
                pcs_rows,
                date(2026, 5, 5),
            )

        self.assertEqual(len(sold_put_rows), 1)
        self.assertEqual(len(pcs_rows), 1)
        self.assertEqual(sold_put_rows[0]["ib_weekly_checked"], "sold_put")
        self.assertEqual(pcs_rows[0]["ib_weekly_checked"], "pcs")

    def test_theta_ib_enrichment_keeps_pcs_cusp_below_twenty_percent_width_credit(self):
        class FakeIb:
            def reqMarketDataType(self, market_data_type):
                self.market_data_type = market_data_type

        pcs_row = {"symbol": "ABC", "score": 75, "notes": "pcs setup"}
        pcs_rows = [pcs_row]
        weekly_chain = [
            {
                "exchange": "SMART",
                "tradingClass": "ABC",
                "multiplier": "100",
                "expirations": {"20260508", "20260515", "20260619"},
                "strikes": {185, 190, 195},
            }
        ]

        def mark_below_target(_ib, row, _chain, _quote_cache, _reference_date):
            row["option_status"] = "cusp"
            row["best_option"] = {
                "play_type": "pcs",
                "status": "cusp",
                "short_strike": 190.0,
                "long_strike": 185.0,
                "width": 5.0,
                "credit": 0.95,
                "credit_width_ratio": 0.19,
                "credit_width_pct": 19.0,
            }

        with (
            patch.object(master_avwap, "is_daily_data_client_connected", return_value=True),
            patch.object(master_avwap, "_fetch_ib_option_chain_definitions", return_value=weekly_chain),
            patch.object(master_avwap, "_enrich_pcs_row_with_ib_options", side_effect=mark_below_target),
        ):
            master_avwap.enrich_theta_rows_with_ib_option_premiums(
                FakeIb(),
                [],
                pcs_rows,
                date(2026, 5, 5),
            )

        self.assertEqual(len(pcs_rows), 1)
        self.assertEqual(pcs_rows[0]["option_status"], "cusp")
        self.assertFalse(master_avwap._pcs_row_meets_credit_target(pcs_rows[0]))
        self.assertNotIn("theta_filter_reason", pcs_row)

    def test_theta_ib_enrichment_keeps_pcs_below_target_credit_for_monitoring(self):
        class FakeIb:
            def reqMarketDataType(self, market_data_type):
                self.market_data_type = market_data_type

        pcs_row = {"symbol": "ABC", "score": 75, "notes": "pcs setup"}
        pcs_rows = [pcs_row]
        weekly_chain = [
            {
                "exchange": "SMART",
                "tradingClass": "ABC",
                "multiplier": "100",
                "expirations": {"20260508", "20260515", "20260619"},
                "strikes": {185, 190, 195},
            }
        ]

        def mark_below_cusp(_ib, row, _chain, _quote_cache, _reference_date):
            row["option_status"] = "below_target"
            row["best_option"] = {
                "play_type": "pcs",
                "status": "below_target",
                "short_strike": 190.0,
                "long_strike": 185.0,
                "width": 5.0,
                "credit": 0.5,
                "credit_width_ratio": 0.10,
                "credit_width_pct": 10.0,
            }

        with (
            patch.object(master_avwap, "is_daily_data_client_connected", return_value=True),
            patch.object(master_avwap, "_fetch_ib_option_chain_definitions", return_value=weekly_chain),
            patch.object(master_avwap, "_enrich_pcs_row_with_ib_options", side_effect=mark_below_cusp),
        ):
            master_avwap.enrich_theta_rows_with_ib_option_premiums(
                FakeIb(),
                [],
                pcs_rows,
                date(2026, 5, 5),
            )

        self.assertEqual(len(pcs_rows), 1)
        self.assertEqual(pcs_row["option_status"], "below_target")
        self.assertEqual(pcs_row["best_option"]["short_strike"], 190.0)
        self.assertEqual(pcs_row["best_option"]["credit"], 0.5)
        self.assertFalse(pcs_row.get("theta_filter_out", False))
        self.assertNotIn("theta_filter_reason", pcs_row)

    def test_sold_put_below_cusp_quote_is_kept_for_monitoring(self):
        row = {
            "symbol": "ABC",
            "last_close": 105.0,
            "score": 80,
            "supports": [
                {"label": "SMA_50", "level": 100.0, "source": "sma", "distance_atr": 0.5},
                {"label": "CURRENT_AVWAPE", "level": 98.0, "source": "avwape", "distance_atr": 0.7},
                {"label": "PREV_AVWAPE", "level": 96.0, "source": "previous_avwape", "distance_atr": 0.9},
            ],
        }
        chain = {
            "tradingClass": "ABC",
            "multiplier": "100",
            "expirations": {"20260508"},
            "strikes": {100, 99, 98, 97, 96, 95},
        }

        with patch.object(
            master_avwap,
            "_fetch_theta_option_quote_cached",
            return_value={"bid": 0.04, "ask": 0.08},
        ):
            master_avwap._enrich_sold_put_row_with_ib_options(
                object(),
                row,
                chain,
                {},
                date(2026, 5, 5),
            )

        self.assertEqual(row["option_status"], "below_target")
        self.assertIn("strike", row["best_option"])
        self.assertAlmostEqual(row["best_option"]["credit"], 0.04)
        self.assertEqual(row["best_option"]["credit_source"], "bid")

    def test_theta_ib_unavailable_keeps_pcs_support_only_rows(self):
        sold_put_row = {"symbol": "ABC", "score": 80, "notes": "theta setup"}
        pcs_row = {"symbol": "ABC", "score": 75, "notes": "pcs setup"}
        sold_put_rows = [sold_put_row]
        pcs_rows = [pcs_row]

        with patch.object(master_avwap, "is_daily_data_client_connected", return_value=False):
            master_avwap.enrich_theta_rows_with_ib_option_premiums(
                None,
                sold_put_rows,
                pcs_rows,
                date(2026, 5, 5),
            )

        self.assertEqual(len(sold_put_rows), 1)
        self.assertEqual(len(pcs_rows), 1)
        self.assertEqual(sold_put_row["option_status"], "ib_unavailable")
        self.assertEqual(pcs_row["option_status"], "ib_unavailable")
        self.assertFalse(pcs_row.get("theta_filter_out", False))

    def test_master_scan_watchlists_include_master_only_swing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            longs_path = root / "longs.txt"
            shorts_path = root / "shorts.txt"
            swing_longs_path = root / "swinglongs.txt"
            swing_shorts_path = root / "shortswings.txt"
            longs_path.write_text("AAPL\nMSFT\n", encoding="utf-8")
            shorts_path.write_text("TSLA\n", encoding="utf-8")
            swing_longs_path.write_text("NVDA\nAAPL\n", encoding="utf-8")
            swing_shorts_path.write_text("AMD\n", encoding="utf-8")

            with (
                patch.object(master_avwap, "LONGS_FILE", longs_path),
                patch.object(master_avwap, "SHORTS_FILE", shorts_path),
                patch.object(master_avwap, "SWING_LONGS_FILE", swing_longs_path),
                patch.object(master_avwap, "SWING_SHORTS_FILE", swing_shorts_path),
            ):
                long_paths, short_paths, label = master_avwap.resolve_master_scan_watchlist_paths()
                longs = master_avwap.load_tickers_from_paths(long_paths, optional_paths={swing_longs_path})
                shorts = master_avwap.load_tickers_from_paths(short_paths, optional_paths={swing_shorts_path})

            self.assertIn("swing watchlists", label)
            self.assertEqual(longs, ["AAPL", "MSFT", "NVDA"])
            self.assertEqual(shorts, ["TSLA", "AMD"])

    def test_eod_write_gate_allows_final_hour_and_after_close(self):
        self.assertFalse(
            master_avwap.should_update_setup_tracker_now(
                now=datetime(2026, 5, 5, 11, 30),
                window_start="12:00",
                window_end="13:00",
            )
        )
        self.assertTrue(
            master_avwap.should_update_setup_tracker_now(
                now=datetime(2026, 5, 5, 12, 30),
                window_start="12:00",
                window_end="13:00",
            )
        )
        self.assertTrue(
            master_avwap.should_update_setup_tracker_now(
                now=datetime(2026, 5, 5, 14, 30),
                window_start="12:00",
                window_end="13:00",
            )
        )

    def test_favorite_zone_watchlist_writes_wait_until_eod_window(self):
        priority_rows = [
            {
                "symbol": "AAPL",
                "side": "LONG",
                "score": 120,
                "priority_bucket": "favorite_setup",
                "setup_family": "favorite_zone_watch",
                "favorite_zone": "AVWAPE to UPPER_1",
                "favorite_signals": [],
                "context_signals": [],
            }
        ]
        ai_state = {"symbols": {"AAPL": {"last_close": 101.0}}}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            focus_path = root / "focus.json"
            d1_path = root / "d1.json"
            focus_path.write_text("existing focus", encoding="utf-8")
            d1_path.write_text("existing d1", encoding="utf-8")

            skipped = master_avwap.write_favorite_zone_watchlist_outputs(
                focus_path=focus_path,
                d1_watchlist_path=d1_path,
                priority_rows=priority_rows,
                theta_put_rows=[],
                theta_pcs_rows=[],
                ai_state=ai_state,
                now=datetime(2026, 5, 5, 11, 30),
                window_start="12:00",
                window_end="13:00",
            )

            self.assertFalse(skipped["updated"])
            self.assertEqual(focus_path.read_text(encoding="utf-8"), "existing focus")
            self.assertEqual(d1_path.read_text(encoding="utf-8"), "existing d1")

            written = master_avwap.write_favorite_zone_watchlist_outputs(
                focus_path=focus_path,
                d1_watchlist_path=d1_path,
                priority_rows=priority_rows,
                theta_put_rows=[],
                theta_pcs_rows=[],
                ai_state=ai_state,
                now=datetime(2026, 5, 5, 14, 30),
                window_start="12:00",
                window_end="13:00",
            )

            self.assertTrue(written["updated"])
            focus_payload = master_avwap.load_json(focus_path, default={})
            d1_payload = master_avwap.load_json(d1_path, default={})
            self.assertEqual(focus_payload["favorites"][0]["symbol"], "AAPL")
            self.assertIn("AAPL", d1_payload["symbols"])

    def test_d1_watchlist_prunes_side_aware_ema15_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "master_avwap_d1_watchlist.json"
            priority_rows = [
                {"symbol": "AAPL", "side": "LONG", "priority_bucket": "favorite_setup", "score": 250},
                {"symbol": "NVDA", "side": "LONG", "priority_bucket": "favorite_setup", "score": 240},
                {"symbol": "TSLA", "side": "SHORT", "priority_bucket": "favorite_setup", "score": 230},
                {"symbol": "AMD", "side": "SHORT", "priority_bucket": "favorite_setup", "score": 220},
            ]
            ai_state = {
                "symbols": {
                    "AAPL": {"last_close": 99.0, "entry_feature_snapshot": {"ema15": 100.0}},
                    "NVDA": {"last_close": 101.0, "entry_feature_snapshot": {"ema15": 100.0}},
                    "TSLA": {"last_close": 101.0, "entry_feature_snapshot": {"ema15": 100.0}},
                    "AMD": {"last_close": 99.0, "entry_feature_snapshot": {"ema15": 100.0}},
                }
            }

            payload = update_master_avwap_d1_watchlist(path, priority_rows, [], [], ai_state)
            self.assertEqual(set(payload["symbols"]), {"NVDA", "AMD"})
            removed = {row["symbol"]: row for row in payload["removed_symbols"]}
            self.assertEqual(set(removed), {"AAPL", "TSLA"})
            self.assertIn("below 15EMA", removed["AAPL"]["note"])
            self.assertIn("above 15EMA", removed["TSLA"]["note"])

    def test_combined_avwap_output_places_theta_near_top(self):
        text = build_combined_avwap_output_text(
            "Priority rows",
            "1. NVDA | close=100.00 | score=10",
            "Event rows",
            "Stdev rows",
        )

        self.assertLess(text.index("MASTER AVWAP THETA PLAYS"), text.index("MASTER AVWAP EVENT TICKERS"))
        self.assertIn("1. NVDA", text)
        self.assertEqual(extract_theta_symbols_from_report(text), ["NVDA"])

    def test_theta_report_and_reason_risk_parser_include_compact_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "theta.txt"
            write_theta_put_report(
                path,
                [
                    {
                        "symbol": "NVDA",
                        "last_close": 100.0,
                        "score": 88,
                        "support_count": 5,
                        "premium_target": "2.00-3.00",
                        "strike_zone": "at/below CURRENT_AVWAPE 99.00",
                        "support_summary": "CURRENT_AVWAPE@99.00",
                        "last_earnings_date": "2026-03-01",
                        "days_since_last_earnings": 50,
                        "next_earnings_date": "2026-06-20",
                        "days_to_next_earnings": 61,
                        "top_score_drivers": "AVWAPE+Trendline confluence",
                        "risk_flags": ["support distance high", "earnings soon"],
                        "notes": "manual IV/premium check",
                    }
                ],
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("reason=AVWAPE+Trendline confluence", text)
            self.assertIn("risk=support distance high, earnings soon", text)
            parsed = extract_theta_reason_risk_rows(text)
            self.assertEqual(parsed[0]["symbol"], "NVDA")
            self.assertIn("AVWAPE+Trendline", parsed[0]["reason"])

    def test_theta_report_parser_surfaces_sell_strike_and_approx_credit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "theta.txt"
            write_theta_put_report(
                path,
                [
                    {
                        "symbol": "NVDA",
                        "last_close": 100.0,
                        "score": 88,
                        "support_count": 5,
                        "strike_zone": "at/below CURRENT_AVWAPE 99.00",
                        "support_summary": "CURRENT_AVWAPE@99.00",
                        "option_status": "recommended",
                        "best_option": {
                            "status": "recommended",
                            "expiration": "20260508",
                            "market_days": 3,
                            "strike": 95.0,
                            "credit": 0.35,
                            "credit_source": "last",
                            "last": 0.35,
                            "contracts_needed_for_100": 3,
                            "total_credit_at_contract_cap": 105.0,
                            "covered_support_count": 3,
                            "total_support_count": 4,
                        },
                    }
                ],
                [
                    {
                        "symbol": "MSFT",
                        "last_close": 200.0,
                        "score": 92,
                        "support_count": 3,
                        "strike_zone": "near AVWAPE 190.00",
                        "support_summary": "AVWAPE@190.00",
                        "option_status": "recommended",
                        "best_option": {
                            "status": "recommended",
                            "expiration": "20260522",
                            "market_days": 14,
                            "short_strike": 190.0,
                            "long_strike": 185.0,
                            "width": 5.0,
                            "credit": 1.0,
                            "credit_source": "bid/ask",
                            "credit_width_ratio": 0.20,
                            "credit_width_pct": 20.0,
                            "max_loss": 400.0,
                            "covered_support_count": 2,
                            "total_support_count": 3,
                        },
                    }
                ],
            )

            text = path.read_text(encoding="utf-8")
            rows = {row["symbol"]: row for row in extract_theta_rows_from_report(text)}

            self.assertIn("@ approx_credit=0.35 (source=last", text)
            self.assertIn("@ approx_credit=1.00 (source=bid/ask", text)
            self.assertEqual(rows["NVDA"]["recommended_strike"], 95.0)
            self.assertEqual(rows["NVDA"]["recommended_credit"], 0.35)
            self.assertEqual(rows["NVDA"]["recommended_credit_source"], "last")
            self.assertEqual(rows["MSFT"]["recommended_strike"], 190.0)
            self.assertEqual(rows["MSFT"]["recommended_long_strike"], 185.0)
            self.assertEqual(rows["MSFT"]["recommended_credit"], 1.0)
            self.assertEqual(rows["MSFT"]["recommended_credit_source"], "bid/ask")

    def test_theta_report_surfaces_pcs_support_only_candidates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "theta.txt"
            write_theta_put_report(
                path,
                [],
                [
                    {
                        "symbol": "MSFT",
                        "last_close": 200.0,
                        "score": 92,
                        "support_count": 2,
                        "strike_zone": "near AVWAPE 190.00",
                        "support_summary": "AVWAPE@190.00, SMA_50@188.00",
                        "option_status": "no_quote",
                        "top_score_drivers": "2 support stack; PCS 2-support minimum",
                        "risk_flags": ["none"],
                        "notes": "IBKR option premium check pending",
                    }
                ],
            )

            text = path.read_text(encoding="utf-8")
            rows = extract_theta_rows_from_report(text)

            self.assertIn("PCS Support-Only / No Quote", text)
            self.assertIn("1. MSFT | close=200.00", text)
            self.assertEqual(rows[0]["symbol"], "MSFT")
            self.assertEqual(rows[0]["play_type"], "pcs")

    def test_theta_reason_risk_parser_supports_legacy_notes_only_reports(self):
        text = "\n".join(
            [
                "1. NVDA | close=100.00 | score=80",
                "   notes=legacy-only theta note",
            ]
        )
        parsed = extract_theta_reason_risk_rows(text)
        self.assertEqual(parsed[0]["symbol"], "NVDA")
        self.assertEqual(parsed[0]["reason"], "legacy-only theta note")
        self.assertIn("legacy report", parsed[0]["risk"])

    def test_previous_avwape_bounce_adds_large_priority_bonus(self):
        base_summary = build_priority_setup_summary(
            symbol="LAUR",
            side="LONG",
            events_today=[],
            all_events=[],
            trend_label="UP",
            favorite_zone=None,
        )
        bounce_summary = build_priority_setup_summary(
            symbol="LAUR",
            side="LONG",
            events_today=["PREV_BOUNCE_VWAP"],
            all_events=["PREV_BOUNCE_VWAP"],
            trend_label="UP",
            favorite_zone=None,
        )

        self.assertTrue(bounce_summary["previous_avwape_bounce"])
        self.assertEqual(
            bounce_summary["previous_avwape_bounce_bonus"],
            master_avwap.PRIORITY_PREVIOUS_AVWAPE_BOUNCE_SCORE_BONUS,
        )
        self.assertEqual(bounce_summary["setup_family"], "previous_avwape_bounce")
        self.assertIn("PREV_BOUNCE_VWAP", bounce_summary["bounce_support_signals"])
        self.assertGreaterEqual(
            bounce_summary["score"] - base_summary["score"],
            master_avwap.PRIORITY_PREVIOUS_AVWAPE_BOUNCE_SCORE_BONUS,
        )

    def test_priority_obstacles_heavily_penalize_previous_critical_sr_ahead(self):
        summary = master_avwap.assess_priority_directional_obstacles(
            side="LONG",
            last_close=100.0,
            atr20=2.0,
            daily_rows=[
                {
                    "date": "2026-05-12",
                    "open": 99.0,
                    "high": 100.4,
                    "low": 98.8,
                    "close": 100.0,
                }
            ],
            last_trade_date="2026-05-12",
            sma_levels={},
            current_anchor_meta=None,
            previous_anchor_meta={
                "vwap": 100.4,
                "bands": {
                    "UPPER_1": 101.0,
                    "LOWER_1": 97.0,
                    "UPPER_2": 102.0,
                    "LOWER_2": 95.0,
                    "UPPER_3": 106.0,
                    "LOWER_3": 93.0,
                },
            },
        )

        labels = {entry["label"] for entry in summary["previous_anchor_obstacles"]}
        critical_labels = {entry["label"] for entry in summary["previous_anchor_critical_obstacles"]}
        measurement_labels = {entry["label"] for entry in summary["previous_anchor_measurement_obstacles"]}

        self.assertFalse(summary["ranking_blocked"])
        self.assertTrue(summary["previous_anchor_critical_obstacle"])
        self.assertIn("PREV_AVWAPE", critical_labels)
        self.assertIn("PREV_UPPER_1", critical_labels)
        self.assertIn("PREV_UPPER_2", measurement_labels)
        self.assertIn("PREV_UPPER_2", labels)
        self.assertGreaterEqual(
            summary["score_penalty_previous_anchor"],
            master_avwap.PRIORITY_PREVIOUS_CRITICAL_SR_NEAR_PENALTY,
        )
        self.assertIn("Critical previous S/R ahead", summary["ranking_note"])

    def test_tracker_attributes_record_previous_anchor_sr_research_levels(self):
        sr_summary = master_avwap.analyze_previous_anchor_sr_context(
            [
                {
                    "date": "2026-05-12",
                    "open": 106.5,
                    "high": 108.2,
                    "low": 105.7,
                    "close": 107.8,
                }
            ],
            "2026-05-12",
            107.8,
            "LONG",
            {
                "vwap": 100.0,
                "bands": {
                    "UPPER_1": 104.0,
                    "LOWER_1": 96.0,
                    "UPPER_2": 106.0,
                    "LOWER_2": 94.0,
                    "UPPER_3": 108.0,
                    "LOWER_3": 92.0,
                },
            },
            4.0,
            events_today=["PREV_BOUNCE_VWAP", "PREV_BOUNCE_UPPER_3"],
        )
        attributes, registry = master_avwap.build_tracker_entry_attributes(
            {
                "side": "LONG",
                "symbol": "LAUR",
                "score": 100,
                "previous_avwape_bounce": True,
            },
            {
                "side": "LONG",
                "symbol": "LAUR",
                "events_today": ["PREV_BOUNCE_VWAP", "PREV_BOUNCE_UPPER_3"],
                "previous_anchor_sr_summary": sr_summary,
                "previous_anchor_near_levels": sr_summary["near_levels"],
                "previous_anchor_touched_levels": sr_summary["touched_levels"],
                "previous_anchor_bounce_levels": sr_summary["bounce_levels"],
                "previous_anchor_critical_near_levels": sr_summary["critical_near_levels"],
                "previous_anchor_critical_touched_levels": sr_summary["critical_touched_levels"],
                "previous_anchor_critical_bounce_levels": sr_summary["critical_bounce_levels"],
                "previous_anchor_critical_obstacle_levels": sr_summary["critical_obstacle_levels"],
                "previous_avwape_bounce": True,
            },
            {
                "previous_anchor_sr_summary": sr_summary,
            },
        )

        self.assertIn("PREV_UPPER_2", attributes["sr.previous_anchor_near_levels"])
        self.assertIn("PREV_UPPER_3", attributes["sr.previous_anchor_near_levels"])
        self.assertTrue(attributes["sr.previous_upper_2_near"])
        self.assertTrue(attributes["sr.previous_upper_3_near"])
        self.assertTrue(attributes["sr.previous_upper_3_bounce"])
        self.assertTrue(attributes["sr.previous_avwape_bounce"])
        self.assertIn("sr.previous_upper_3_bounce", registry)

    def test_previous_day_range_break_tracks_intraday_hod_tag_without_close_confirm(self):
        rows = [
            {"date": "2026-05-11", "open": 98.0, "high": 100.0, "low": 97.0, "close": 99.0},
            {"date": "2026-05-12", "open": 99.0, "high": 101.0, "low": 98.5, "close": 99.5},
        ]

        result = master_avwap.assess_previous_day_range_break(
            rows,
            date(2026, 5, 12),
            99.5,
            "LONG",
        )

        self.assertTrue(result["previous_day_range_intraday_break"])
        self.assertFalse(result["previous_day_range_break"])
        self.assertEqual(result["previous_day_range_break_bonus"], 0)
        self.assertIn("Tagged previous day high", result["previous_day_range_intraday_note"])

    def test_adverse_entry_candle_detects_long_upper_wick_close_near_low(self):
        result = master_avwap.assess_adverse_entry_candle(
            [
                {
                    "date": "2026-05-12",
                    "open": 100.0,
                    "high": 110.0,
                    "low": 98.0,
                    "close": 99.0,
                }
            ],
            "2026-05-12",
            "LONG",
            atr20=10.0,
        )

        self.assertTrue(result["adverse_entry_candle"])
        self.assertIn("upper wick rejection", result["adverse_entry_candle_note"])

    def test_high_conviction_rejects_favorite_zone_without_real_trigger(self):
        row = {
            "symbol": "DB",
            "side": "LONG",
            "score": 140,
            "priority_bucket": "favorite_setup",
            "setup_family": "favorite_zone_watch",
            "favorite_signals": [],
            "favorite_zone": "AVWAPE to UPPER_1",
            "current_band_zone": "VWAP to UPPER_1",
            "previous_day_range_intraday_break": True,
        }

        self.assertFalse(master_avwap._priority_is_high_conviction(row))

    def test_high_conviction_accepts_clean_intraday_hod_break_with_avwap_trigger(self):
        row = {
            "symbol": "HBAN",
            "side": "LONG",
            "score": 140,
            "priority_bucket": "favorite_setup",
            "setup_family": "avwap_breakout",
            "favorite_signals": ["CROSS_UP_VWAP"],
            "favorite_zone": "AVWAPE to UPPER_1",
            "current_band_zone": "VWAP to UPPER_1",
            "previous_day_range_intraday_break": True,
        }

        self.assertTrue(master_avwap._priority_is_high_conviction(row))

    def test_high_conviction_rejects_adverse_rejection_candle(self):
        row = {
            "symbol": "BCRX",
            "side": "LONG",
            "score": 140,
            "priority_bucket": "favorite_setup",
            "setup_family": "avwap_breakout",
            "favorite_signals": ["CROSS_UP_VWAP"],
            "favorite_zone": "AVWAPE to UPPER_1",
            "current_band_zone": "VWAP to UPPER_1",
            "previous_day_range_intraday_break": True,
            "adverse_entry_candle": True,
            "adverse_entry_candle_note": "big upper wick rejection",
        }

        self.assertFalse(master_avwap._priority_is_high_conviction(row))

    def test_high_conviction_rejects_compressed_dual_first_dev_resistance_without_close_break(self):
        row = {
            "symbol": "ELAN",
            "side": "LONG",
            "score": 160,
            "priority_bucket": "favorite_setup",
            "setup_family": "avwap_breakout",
            "favorite_signals": ["CROSS_UP_VWAP"],
            "favorite_zone": "AVWAPE to UPPER_1",
            "current_band_zone": "VWAP to UPPER_1",
            "previous_day_range_intraday_break": True,
            "previous_day_range_break": False,
            "compression_flag": True,
            "compression_penalty": master_avwap.PRIORITY_COMPRESSION_SCORE_PENALTY_SEVERE,
            "current_first_dev_obstacle": True,
            "previous_anchor_critical_obstacle": True,
        }

        self.assertFalse(master_avwap._priority_is_high_conviction(row))

    def test_priority_report_leads_with_sab_tiers_before_setup_types(self):
        rows = [
            {
                "symbol": "AAA",
                "side": "LONG",
                "score": 220,
                "priority_bucket": "favorite_setup",
                "setup_family": "avwap_breakout",
                "favorite_signals": ["CROSS_UP_VWAP"],
                "current_band_zone": "VWAP to UPPER_1",
                "previous_day_range_break": True,
                "previous_day_range_intraday_break": True,
                "vwap_range_confirmation": True,
                "trend_20d": "UP",
            },
            {
                "symbol": "BBB",
                "side": "LONG",
                "score": 190,
                "priority_bucket": "favorite_setup",
                "setup_family": "favorite_zone_watch",
                "favorite_zone": "AVWAPE to UPPER_1",
                "current_band_zone": "VWAP to UPPER_1",
                "trend_20d": "UP",
            },
            {
                "symbol": "CCC",
                "side": "LONG",
                "score": 170,
                "priority_bucket": "near_favorite_zone",
                "setup_family": "favorite_zone_watch",
                "favorite_zone": "AVWAPE to UPPER_1",
                "current_band_zone": "VWAP to UPPER_1",
                "trend_20d": "UP",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "priority.txt"
            write_priority_setup_report(report_path, rows)
            report = report_path.read_text(encoding="utf-8")

        cream_copy = report.split("Cream of the crop details", 1)[0]

        self.assertLess(report.index("Cream of the crop copy/paste"), report.index("Copy/paste lists"))
        self.assertLess(report.index("Cream of the crop details"), report.index("Copy/paste lists"))
        self.assertIn("S Tier - act first\n  LONG: AAA", cream_copy)
        self.assertIn("A Tier - next best\n  LONG: BBB", cream_copy)
        self.assertIn("B Tier - stalk list\n  LONG: CCC", cream_copy)
        self.assertLess(report.index("Detailed setup notes"), report.index("Appendix: setup-type copy lists"))
        self.assertLess(report.index("Appendix: setup-type copy lists"), report.index("By setup type"))


if __name__ == "__main__":
    unittest.main()
