"""Tests for the weekly-8EMA persistence basket study family.

Basket = long names whose last WEEKLY_EMA8_HOLD_MIN_WEEKS weekly candles all
closed at/above the weekly 8EMA. Membership is flagged on the symbol entry every
scan; a study row is recorded only when today's daily bar tags a bounce target
(daily EMA15 / EMA21 / current-anchor first deviation) and closes back above it.
Recorded in the isolated study namespace so realized R accrues before any
promotion to a scored family.
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


def _uptrend_frame(periods: int = 260, start: float = 50.0, step: float = 0.5) -> pd.DataFrame:
    """Steadily rising dailies -> weekly closes sit above the (lagging) weekly EMA8."""
    dates = pd.bdate_range("2025-07-07", periods=periods)
    rows = []
    for i, dt in enumerate(dates):
        base = start + i * step
        rows.append(
            {
                "datetime": dt,
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.3,
                "volume": 1_000_000,
            }
        )
    return pd.DataFrame(rows)


def _downtrend_frame(periods: int = 260, start: float = 200.0, step: float = 0.5) -> pd.DataFrame:
    dates = pd.bdate_range("2025-07-07", periods=periods)
    rows = []
    for i, dt in enumerate(dates):
        base = start - i * step
        rows.append(
            {
                "datetime": dt,
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base - 0.3,
                "volume": 1_000_000,
            }
        )
    return pd.DataFrame(rows)


def _scan_inputs(symbol, df, *, ema15, low, close, atr20=2.0, bands=None):
    last_date = df.iloc[-1]["datetime"].date().isoformat()
    today = {"date": last_date, "open": close, "high": close + 1.0, "low": low, "close": close, "volume": 1_000_000}
    priority_rows = [
        {
            "symbol": symbol,
            "side": "LONG",
            "last_trade_date": last_date,
            "atr20": atr20,
            "ema_15": ema15,
            "ema_21": ema15 - 2.0,
        }
    ]
    ai_state = {
        "symbols": {
            symbol: {
                "side": "LONG",
                "daily_ohlc": [today],
                "atr20": atr20,
                "last_trade_date": last_date,
                "current_anchor": {"bands": dict(bands or {})},
            }
        }
    }
    return priority_rows, ai_state


class WeeklyEma8HoldTests(unittest.TestCase):
    def test_basket_member_with_ema15_bounce_emits_study_row(self):
        df = _uptrend_frame()
        last_close = float(df.iloc[-1]["close"])
        ema15 = last_close - 2.0
        priority_rows, ai_state = _scan_inputs(
            "HOLD", df, ema15=ema15, low=ema15 - 0.1, close=last_close
        )
        study = m.enrich_priority_rows_with_weekly_ema8_hold(
            priority_rows, {"HOLD": df}, ai_state=ai_state
        )
        self.assertEqual(len(study), 1)
        row = study[0]
        self.assertEqual(row["setup_family"], m.WEEKLY_EMA8_HOLD_STUDY_FAMILY)
        self.assertEqual(row["priority_bucket"], m.WEEKLY_EMA8_HOLD_STUDY_BUCKET)
        self.assertEqual(row["weekly_ema8_bounce_level_label"], "EMA_15")
        self.assertIn("WEEKLY_EMA8_HOLD", row["setup_tags"])
        self.assertFalse(row["is_favorite_setup"])
        self.assertGreaterEqual(row["weekly_ema8_hold_weeks"], m.WEEKLY_EMA8_HOLD_MIN_WEEKS)
        # Basket membership is flagged on the symbol entry too.
        self.assertTrue(ai_state["symbols"]["HOLD"]["weekly_ema8_hold"])

    def test_basket_member_without_touch_flags_membership_only(self):
        df = _uptrend_frame()
        last_close = float(df.iloc[-1]["close"])
        ema15 = last_close - 10.0  # far below today's low -> no tag
        priority_rows, ai_state = _scan_inputs(
            "FLAG", df, ema15=ema15, low=last_close - 1.0, close=last_close
        )
        study = m.enrich_priority_rows_with_weekly_ema8_hold(
            priority_rows, {"FLAG": df}, ai_state=ai_state
        )
        self.assertEqual(study, [])
        self.assertTrue(ai_state["symbols"]["FLAG"]["weekly_ema8_hold"])
        self.assertGreaterEqual(
            ai_state["symbols"]["FLAG"]["weekly_ema8_hold_weeks"], m.WEEKLY_EMA8_HOLD_MIN_WEEKS
        )

    def test_downtrend_is_not_in_basket(self):
        df = _downtrend_frame()
        last_close = float(df.iloc[-1]["close"])
        priority_rows, ai_state = _scan_inputs(
            "DOWN", df, ema15=last_close + 1.0, low=last_close - 1.0, close=last_close
        )
        study = m.enrich_priority_rows_with_weekly_ema8_hold(
            priority_rows, {"DOWN": df}, ai_state=ai_state
        )
        self.assertEqual(study, [])
        self.assertFalse(ai_state["symbols"]["DOWN"]["weekly_ema8_hold"])

    def test_first_dev_bounce_used_when_emas_not_tagged(self):
        df = _uptrend_frame()
        last_close = float(df.iloc[-1]["close"])
        first_dev = last_close - 2.0
        priority_rows, ai_state = _scan_inputs(
            "BAND",
            df,
            ema15=last_close - 10.0,  # not tagged
            low=first_dev - 0.1,
            close=last_close,
            bands={"UPPER_1": first_dev},
        )
        # EMA21 sits at ema15-2 = last_close-12 -> low (first_dev-0.1 = last_close-2.1)
        # does not reach it, so only UPPER_1 tags.
        study = m.enrich_priority_rows_with_weekly_ema8_hold(
            priority_rows, {"BAND": df}, ai_state=ai_state
        )
        self.assertEqual(len(study), 1)
        self.assertEqual(study[0]["weekly_ema8_bounce_level_label"], "UPPER_1")

    def test_short_side_and_missing_frame_are_skipped(self):
        df = _uptrend_frame()
        priority_rows, ai_state = _scan_inputs("SKIP", df, ema15=100.0, low=99.0, close=101.0)
        priority_rows[0]["side"] = "SHORT"
        ai_state["symbols"]["SKIP"]["side"] = "SHORT"
        self.assertEqual(
            m.enrich_priority_rows_with_weekly_ema8_hold(priority_rows, {"SKIP": df}, ai_state=ai_state),
            [],
        )
        # LONG but no daily frame available -> skipped without crash.
        priority_rows2, ai_state2 = _scan_inputs("NODF", df, ema15=100.0, low=99.0, close=101.0)
        self.assertEqual(
            m.enrich_priority_rows_with_weekly_ema8_hold(priority_rows2, {}, ai_state=ai_state2),
            [],
        )


class WeeklyEma8StreakTests(unittest.TestCase):
    def test_streak_counts_consecutive_closes_above_ema8(self):
        weekly = m.compute_weekly_indicator_frame(_uptrend_frame())
        self.assertIn("ema_8", weekly.columns)
        streak = m._weekly_ema8_hold_streak(weekly)
        self.assertGreaterEqual(streak, m.WEEKLY_EMA8_HOLD_MIN_WEEKS)

    def test_streak_breaks_on_close_below_ema8(self):
        df = _uptrend_frame()
        weekly = m.compute_weekly_indicator_frame(df)
        # Force the 3rd-from-last weekly close under its EMA8 -> streak resets to 2.
        weekly = weekly.copy()
        idx = len(weekly) - 3
        weekly.loc[idx, "close_num"] = float(weekly.loc[idx, "ema_8"]) - 1.0
        self.assertEqual(m._weekly_ema8_hold_streak(weekly), 2)


if __name__ == "__main__":
    unittest.main()
