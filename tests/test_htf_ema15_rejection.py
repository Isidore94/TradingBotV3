"""Tests for the H1/H4 15EMA rejection study family + session-aligned 4h bars.

A bar that pierces the 15EMA intrabar and closes back on the trend side, while
both HTF trends align with the setup side, is recorded as the isolated
``htf_ema15_rejection`` study family (no scoring impact) so realized R accrues
before any promotion. The 4h resample restarts its 4-bar grouping at each
session boundary so 4h EMAs line up with chart-drawn session-anchored candles.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402


def _hourly_uptrend(hours: int = 360) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def _with_last_bar_ema15_dip(frame: pd.DataFrame) -> pd.DataFrame:
    indicators = m.compute_indicator_frame(frame)
    ema15 = float(indicators.iloc[-1]["ema_15"])
    work = frame.copy()
    work.loc[work.index[-1], "low"] = ema15 - 0.10  # pierce; close stays above
    return work


class SessionAligned4hResampleTests(unittest.TestCase):
    def test_grouping_restarts_at_each_session(self):
        # Two 7-bar RTH sessions -> ceil(7/4) = 2 groups per day, 4 bars total.
        rows = []
        price = 100.0
        for day in ("2026-06-01", "2026-06-02"):
            for hour in range(7):
                rows.append(
                    {
                        "datetime": pd.Timestamp(f"{day} 09:30") + pd.Timedelta(hours=hour),
                        "open": price,
                        "high": price + 0.5,
                        "low": price - 0.5,
                        "close": price + 0.2,
                        "volume": 100,
                    }
                )
                price += 1.0
        frame = pd.DataFrame(rows)
        four_hour = m.resample_intraday_bars_to_4h(frame)

        self.assertEqual(len(four_hour), 4)
        # Day 1's second group holds only its last 3 bars (no bleed into day 2).
        self.assertEqual(float(four_hour.iloc[1]["volume"]), 300.0)
        # Day 2's first 4h bar opens with day 2's first 1h bar.
        day2_first_open = float(frame.iloc[7]["open"])
        self.assertEqual(float(four_hour.iloc[2]["open"]), day2_first_open)
        self.assertEqual(four_hour.iloc[2]["datetime"].date().isoformat(), "2026-06-02")


class HtfEma15RejectionTests(unittest.TestCase):
    def test_rejection_detected_and_confirmed_in_aligned_uptrend(self):
        frame = _with_last_bar_ema15_dip(_hourly_uptrend())
        context = m.assess_htf_trend_context(frame, "LONG")
        self.assertEqual(context["htf_trend_1h"], "UP")
        self.assertTrue(context["htf_trend_aligned"])
        self.assertTrue(context["htf_ema15_rejection_confirmed"])
        self.assertIn("1h", context["htf_ema15_rejection_timeframes"])
        self.assertEqual(context["htf_ema15_rejection_age_bars"], 0)
        self.assertIsNotNone(context["htf_ema15_rejection_level"])

    def test_no_rejection_without_a_pierce(self):
        # Steady uptrend: hourly lows never reach the (lagging) 15EMA.
        context = m.assess_htf_trend_context(_hourly_uptrend(), "LONG")
        self.assertTrue(context["htf_trend_aligned"])
        self.assertFalse(context["htf_ema15_rejection_confirmed"])
        self.assertEqual(context["htf_ema15_rejection_timeframes"], "")

    def test_close_through_ema15_is_not_a_rejection(self):
        frame = _hourly_uptrend()
        indicators = m.compute_indicator_frame(frame)
        ema15 = float(indicators.iloc[-1]["ema_15"])
        frame.loc[frame.index[-1], "low"] = ema15 - 2.0
        frame.loc[frame.index[-1], "close"] = ema15 - 1.0  # closed through, no reclaim
        context = m.assess_htf_trend_context(frame, "LONG")
        self.assertFalse(context["htf_ema15_rejection_confirmed"])

    def test_enrich_emits_ema15_study_row(self):
        frame = _with_last_bar_ema15_dip(_hourly_uptrend())
        priority_rows = [
            {"symbol": "REJ", "side": "LONG", "has_favorite_signal": True, "score": 100}
        ]
        study = m.enrich_priority_rows_with_htf_trend_context(
            priority_rows,
            ib=None,
            intraday_frames_by_symbol={"REJ": frame},
            scoring_enabled=False,
        )
        ema15_rows = [r for r in study if r["setup_family"] == m.HTF_EMA15_REJECTION_STUDY_FAMILY]
        self.assertEqual(len(ema15_rows), 1)
        row = ema15_rows[0]
        self.assertEqual(row["priority_bucket"], m.HTF_EMA15_REJECTION_STUDY_BUCKET)
        self.assertIn("HTF_EMA15_REJECTION", row["setup_tags"])
        self.assertFalse(row["is_favorite_setup"])
        # The rejection fields also land on the priority row for report visibility.
        self.assertTrue(priority_rows[0]["htf_ema15_rejection_confirmed"])


if __name__ == "__main__":
    unittest.main()
