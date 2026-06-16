import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
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
    append_d1_feature_history,
    append_master_avwap_user_favorites,
    apply_final_priority_buckets,
    apply_priority_rejection_score_caps,
    apply_tracker_setup_type_adjustments,
    apply_recent_tracker_setup_family_adjustments,
    attach_setup_candidate_payloads,
    build_market_prep_payload,
    build_combined_avwap_output_text,
    build_priority_setup_summary,
    build_master_avwap_focus_setup_type_text,
    build_master_avwap_focus_side_groups,
    build_recent_tracker_setup_family_rows,
    compute_indicator_frame,
    evaluate_theta_put_candidate,
    extract_theta_reason_risk_rows,
    extract_theta_symbols_from_report,
    format_market_prep_payload_report,
    load_scan_earnings_context,
    rank_tracker_setup_type_rows,
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
                close_price = 118.7
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
                close_price = 81.3
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
        self.assertIn("latest close reclaimed SMA_200", result["note"])

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

        detail_start = text.index("Best swing trades today", text.index("By setup type"))
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
            "current_band_zone": "UPPER_2 to UPPER_3",
            "trend_20d": "UP",
            "current_day_open": 100.0,
            "last_close": 104.0,
            "intraday_vwap": 101.0,
        }
        ai_state = {"symbols": {"MIDB": {"current_band_zone": "UPPER_2 to UPPER_3"}}}

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
            "current_band_zone": "VWAP to UPPER_1",
            "current_day_open": 99.0,
            "last_close": 101.0,
            "intraday_vwap": 100.0,
        }
        ai_state = {"symbols": {"FDEV": {"current_band_zone": "VWAP to UPPER_1"}}}

        apply_final_priority_buckets([row], ai_state, [], {})

        self.assertEqual(row["priority_bucket"], "favorite_setup")
        self.assertTrue(row["is_favorite_setup"])

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
                    "symbol": "OLD",
                    "side": "LONG",
                    "score": 300,
                    "post_earnings_active": True,
                    "post_earnings_sessions_since_gap": 11,
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

    def test_post_earnings_52w_break_uses_earnings_candle_high_and_fresh_break(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=0)
        context = _build_post_earnings_52w_release_context(df)

        summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertTrue(summary["qualified_gap"])
        self.assertEqual(summary["monitor_level"], 120.0)
        self.assertEqual(summary["monitor_level_label"], "52W_HIGH")
        self.assertTrue(summary["break_signal"])
        self.assertTrue(summary["break_fresh"])
        self.assertEqual(summary["break_age_sessions"], 0)
        self.assertIn(master_avwap.POST_EARNINGS_BREAK_SIGNAL, summary["events"])

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

    def test_post_earnings_setup_requires_past_10_market_days(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=10)
        context = _build_post_earnings_52w_release_context(df)
        self.assertEqual(context["sessions_since_gap"], 11)
        self.assertTrue(context["in_post_earnings_window"])

        summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertFalse(summary["active"])
        self.assertFalse(summary["qualified_gap"])
        self.assertFalse(summary["break_signal"])
        self.assertEqual(summary["events"], [])

    def test_stale_post_earnings_52w_break_can_still_be_avwape_retest(self):
        df = _build_post_earnings_52w_history(extra_sessions_after_break=4)
        context = _build_post_earnings_52w_release_context(df)

        with patch("master_avwap.bounce_up_at_level", return_value=True):
            summary = master_avwap.analyze_post_earnings_setups(df, "LONG", context)

        self.assertFalse(summary["break_signal"])
        self.assertTrue(summary["bounce_signal"])
        self.assertEqual(summary["family"], master_avwap.POST_EARNINGS_BOUNCE_SIGNAL)
        self.assertEqual(summary["events"], [master_avwap.POST_EARNINGS_BOUNCE_SIGNAL])

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
        self.assertEqual(candidate["side"], "LONG")
        self.assertIn("SMA_50", candidate["major_sma_supports"])
        self.assertTrue(candidate["top_score_drivers"])
        self.assertIsInstance(candidate["risk_flags"], list)

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
        self.assertNotIn(98.0, [candidate["strike"] for candidate in candidates])

        credits = {96.0: 0.30, 94.0: 0.20, 92.0: 0.12}
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
        self.assertEqual(ranked[0]["strike"], 96.0)
        self.assertEqual(ranked[0]["status"], "recommended")
        self.assertEqual(ranked[0]["contracts_needed_for_100"], 4)
        self.assertEqual(ranked[0]["covered_support_count"], 3)

    def test_pcs_ranking_uses_two_supports_and_credit_width_target(self):
        row = {
            "symbol": "NVDA",
            "last_close": 205.0,
            "score": 75,
            "base_score": 75,
            "supports": [
                {"label": "SMA_20", "level": 200.0, "source": "sma", "distance_atr": 0.4},
                {"label": "CURRENT_AVWAPE", "level": 190.0, "source": "avwape", "distance_atr": 1.0},
            ],
        }

        short_candidates = master_avwap._pcs_short_strike_candidates(row, [195, 190, 185])
        self.assertEqual(short_candidates[0]["short_strike"], 190.0)

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


if __name__ == "__main__":
    unittest.main()
