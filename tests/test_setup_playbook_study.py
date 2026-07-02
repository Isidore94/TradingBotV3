"""Tests for the setup playbook study (pure computation, no network/disk)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_playbook_study as study  # noqa: E402


def _frame(closes, *, volume=1_000_000.0, start="2026-01-02", spread=0.5) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    rows = []
    prev = closes[0]
    for dt, close in zip(dates, closes):
        rows.append(
            {
                "datetime": dt,
                "open": prev,
                "high": max(prev, close) + spread,
                "low": min(prev, close) - spread,
                "close": close,
                "volume": volume,
            }
        )
        prev = close
    return pd.DataFrame(rows)


class MeasureEpisodeTests(unittest.TestCase):
    def _arrays(self, frame):
        return (
            frame["open"].to_numpy(float),
            frame["high"].to_numpy(float),
            frame["low"].to_numpy(float),
            frame["close"].to_numpy(float),
            np.full(len(frame), 2.0),  # ATR
        )

    def test_long_stop_first_fills_at_stop(self):
        closes = [100.0] * 5 + [100.0, 101.0, 90.0, 95.0]
        frame = _frame(closes, spread=0.2)
        o, h, l, c, atr = self._arrays(frame)
        result = study.measure_episode(o, h, l, c, atr, 5, "LONG")
        self.assertEqual(result["status"], "STOPPED")
        # stop = signal low - 0.1*ATR; the crash bar trades through it
        self.assertAlmostEqual(result["stop"], l[5] - 0.2, places=6)
        self.assertLess(result["net_r"], 0)
        self.assertGreaterEqual(result["net_r"], -1.5)  # gap-at-open can exceed -1R slightly

    def test_long_time_stop_positive_r(self):
        closes = [100.0 + 0.5 * k for k in range(30)]
        frame = _frame(closes, spread=0.1)
        o, h, l, c, atr = self._arrays(frame)
        result = study.measure_episode(o, h, l, c, atr, 2, "LONG")
        self.assertEqual(result["status"], "TIME_STOP")
        self.assertEqual(result["hold_sessions"], study.TRACKER_MAX_HOLD_DAYS)
        self.assertGreater(result["net_r"], 0)
        self.assertIsNotNone(result["r_5"])
        self.assertGreater(result["r_10"], result["r_5"])  # steady uptrend keeps accruing

    def test_short_mirrors_long_r(self):
        closes = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0] + [90.0] * 15
        frame = _frame(closes, spread=0.1)
        o, h, l, c, atr = self._arrays(frame)
        short = study.measure_episode(o, h, l, c, atr, 2, "SHORT")
        self.assertIsNotNone(short)
        self.assertGreater(short["net_r"], 0)

    def test_untradeable_when_gap_through_stop(self):
        # Next open far below the signal bar's stop -> risk <= 0 -> None.
        closes = [100.0] * 5 + [100.0, 80.0, 80.0]
        frame = _frame(closes, spread=0.2)
        frame.loc[6, "open"] = 80.0  # gap down through the stop before entry
        o, h, l, c, atr = self._arrays(frame)
        self.assertIsNone(study.measure_episode(o, h, l, c, atr, 5, "LONG"))

    def test_no_next_bar_returns_none(self):
        frame = _frame([100.0] * 6, spread=0.2)
        o, h, l, c, atr = self._arrays(frame)
        self.assertIsNone(study.measure_episode(o, h, l, c, atr, 5, "LONG"))


class ContextAndDetectorTests(unittest.TestCase):
    def _uptrend_with_pullback(self, periods=100):
        closes = []
        price = 50.0
        for k in range(periods):
            price *= 1.008
            closes.append(price)
        # carve a one-day pullback near the end that tags EMA8 and recovers
        closes[-3] = closes[-4] * 0.97
        closes[-2] = closes[-4] * 1.01
        closes[-1] = closes[-4] * 1.02
        return closes

    def test_band_context_uses_as_of_anchor(self):
        frame = _frame(self._uptrend_with_pullback(), spread=0.3)
        earnings = [frame["datetime"].iloc[10].date().isoformat()]
        ctx = study.build_symbol_context("TEST", frame, earnings, scan_start_idx=40)
        self.assertTrue(np.isfinite(ctx.vwap[60]))
        self.assertTrue(np.isfinite(ctx.upper1[60]))
        self.assertGreater(ctx.upper2[60], ctx.upper1[60])
        self.assertGreater(ctx.upper1[60], ctx.vwap[60])
        # as-of lookups agree with the arrays on the same day
        self.assertAlmostEqual(ctx.band_asof(60, 60, "UPPER_1"), ctx.upper1[60], places=9)

    def test_mirrored_context_negates_bands(self):
        frame = _frame(self._uptrend_with_pullback(), spread=0.3)
        earnings = [frame["datetime"].iloc[10].date().isoformat()]
        long_ctx = study.build_symbol_context("TEST", frame, earnings, scan_start_idx=40)
        short_ctx = study.build_symbol_context("TEST", frame, earnings, scan_start_idx=40, mirrored=True)
        # mirrored UPPER_1 is the negated original LOWER_1 (dual band)
        self.assertAlmostEqual(short_ctx.vwap[60], -long_ctx.vwap[60], places=9)
        self.assertAlmostEqual(
            short_ctx.band_asof(60, 60, "UPPER_1"), -long_ctx.band_asof(60, 60, "LOWER_1"), places=9
        )

    def test_short_side_detects_mirror_of_long(self):
        closes = self._uptrend_with_pullback()
        up = _frame(closes, spread=0.3)
        down = up.copy()
        for col in ("open", "close"):
            down[col] = 200.0 - up[col]
        down["high"] = 200.0 - up["low"]
        down["low"] = 200.0 - up["high"]
        earnings = [up["datetime"].iloc[10].date().isoformat()]

        long_eps = study.run_symbol("UP", up, earnings, days=40)
        short_eps = study.run_symbol("UP", down, earnings, days=40)
        long_dates = {e["signal_date"] for e in long_eps if e["family"] == "ema8_pullback_uptrend" and e["side"] == "LONG"}
        short_dates = {e["signal_date"] for e in short_eps if e["family"] == "ema8_pullback_uptrend" and e["side"] == "SHORT"}
        self.assertTrue(long_dates)
        self.assertEqual(long_dates, short_dates)

    def test_episode_dedup_blocks_overlapping_signals(self):
        # baseline_every5 fires every 5 sessions but episodes hold up to 19
        # sessions, so consecutive episodes must never overlap.
        closes = [100.0 + 0.1 * k for k in range(140)]
        frame = _frame(closes, spread=0.2)
        episodes = [
            e
            for e in study.run_symbol("FLAT", frame, [], days=60)
            if e["family"] == "baseline_every5" and e["side"] == "LONG"
        ]
        self.assertTrue(episodes)
        entries = sorted(e["entry_date"] for e in episodes)
        for prev, cur in zip(episodes, episodes[1:]):
            self.assertGreater(cur["signal_date"], prev["entry_date"])

    def test_weekly_streak_series_signed_and_no_lookahead(self):
        # 30 weeks up then 10 weeks down, 5 bars per week.
        closes = []
        price = 100.0
        for _week in range(30):
            for _d in range(5):
                price *= 1.01
                closes.append(price)
        for _week in range(10):
            for _d in range(5):
                price *= 0.97
                closes.append(price)
        frame = _frame(closes, start="2025-01-06")
        streaks = study.compute_weekly_streak_series(frame)
        self.assertEqual(len(streaks), len(frame))
        self.assertEqual(streaks[0], 0)  # EMA warmup window
        self.assertGreaterEqual(streaks[149], study.WEEKLY_STREAK_STRONG_WEEKS)  # deep in the uptrend
        self.assertLessEqual(streaks[-1], -study.WEEKLY_STREAK_STRONG_WEEKS)  # deep in the downtrend
        # First day of the collapse still reads the completed-week streak (no lookahead).
        self.assertGreater(streaks[150], 0)
        self.assertEqual(study.classify_weekly_context(int(streaks[149])), "weekly_strong")
        self.assertEqual(study.classify_weekly_context(int(streaks[-1])), "weekly_weak")
        self.assertEqual(study.classify_weekly_context(2), "weekly_mixed")

    def test_episodes_carry_weekly_context_fields(self):
        closes = [100.0 + 0.1 * k for k in range(140)]
        episodes = study.run_symbol("FLAT", _frame(closes, spread=0.2), [], days=40)
        self.assertTrue(episodes)
        self.assertIn("weekly_ctx", episodes[0])
        self.assertIn("weekly_streak", episodes[0])

    def test_ai_digest_verdicts_and_baseline_edges(self):
        def _ep(family, side, ctx, net_r, count):
            return [
                {"symbol": f"S{k}", "family": family, "group": "g", "side": side, "status": "TIME_STOP",
                 "net_r": net_r, "r_5": net_r, "r_10": net_r, "hold_sessions": 10,
                 "weekly_ctx": ctx, "weekly_streak": 6 if ctx == "weekly_strong" else 0,
                 "stop_rate": 0.0, "signal_date": "2026-05-01", "entry_date": "2026-05-02"}
                for k in range(count)
            ]

        episodes = (
            _ep("baseline_every5", "LONG", "weekly_strong", -0.5, 40)
            + _ep("winner_fam", "LONG", "weekly_strong", 0.5, 40)
            + _ep("loser_fam", "LONG", "weekly_strong", -1.0, 40)
            + _ep("thin_fam", "LONG", "weekly_strong", 3.0, 3)
        )
        digest = study.build_ai_digest(episodes, days=60)
        by_key = {(c["family"], c["weekly_ctx"]): c for c in digest["all_combos"]}
        winner = by_key[("winner_fam", "weekly_strong")]
        self.assertEqual(winner["verdict"], "actionable")
        self.assertAlmostEqual(winner["edge_r_vs_baseline"], 1.0, places=6)
        self.assertIn("weekly-strong", winner["action"])
        self.assertEqual(by_key[("loser_fam", "weekly_strong")]["verdict"], "avoid")
        self.assertEqual(by_key[("thin_fam", "weekly_strong")]["verdict"], "insufficient_data")
        self.assertIn("LONG|weekly_strong", digest["baselines"])

    def test_leaderboard_aggregates_and_sorts(self):
        episodes = [
            {"symbol": "A", "family": "fam1", "group": "g", "side": "LONG", "status": "TIME_STOP",
             "net_r": 1.0, "r_5": 0.5, "r_10": 1.0, "hold_sessions": 18},
            {"symbol": "B", "family": "fam1", "group": "g", "side": "LONG", "status": "STOPPED",
             "net_r": -1.0, "r_5": -1.0, "r_10": -1.0, "hold_sessions": 3},
            {"symbol": "C", "family": "fam2", "group": "g", "side": "LONG", "status": "TIME_STOP",
             "net_r": 2.0, "r_5": 1.0, "r_10": 2.0, "hold_sessions": 18},
        ]
        rows = study.aggregate_leaderboard(episodes)
        self.assertEqual(rows[0]["family"], "fam2")
        fam1 = next(r for r in rows if r["family"] == "fam1")
        self.assertEqual(fam1["closed"], 2)
        self.assertAlmostEqual(fam1["avg_r"], 0.0, places=9)
        self.assertAlmostEqual(fam1["win_rate"], 0.5, places=9)
        self.assertAlmostEqual(fam1["stop_rate"], 0.5, places=9)


if __name__ == "__main__":
    unittest.main()
