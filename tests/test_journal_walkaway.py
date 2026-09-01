"""Tests for the walkaway analysis engine (pure computation, no network/DB)."""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_walkaway as jw  # noqa: E402


def _bars(closes: list[float], *, start: str = "2026-03-02", spread: float = 0.5) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    rows = [
        {
            "datetime": dt,
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": 1_000_000,
        }
        for dt, close in zip(dates, closes)
    ]
    return pd.DataFrame(rows)


def _date_at(bars: pd.DataFrame, idx: int) -> str:
    return pd.Timestamp(bars.iloc[idx]["datetime"]).date().isoformat()


class ComputeWalkawayTests(unittest.TestCase):
    def _long_trade_bars(self):
        # 30 flat warmup bars (ATR ~= 1.0 with spread 0.5), then: entry at 100,
        # dip to ~99 (heat), run to ~106 (peak at ~8 bars), settle ~105.
        closes = [100.0] * 30 + [100.0, 99.4, 99.6, 101.0, 102.5, 104.0, 105.5, 106.0, 105.5, 105.0] + [105.0] * 15
        return _bars(closes), 30  # entry index

    def test_long_trade_metrics_and_walkaway(self):
        bars, entry_idx = self._long_trade_bars()
        exit_idx = entry_idx + 4  # sold at 102.5, before most of the move
        position = jw.WalkawayPosition(
            source="journal",
            symbol="TEST",
            side="LONG",
            entry_date=_date_at(bars, entry_idx),
            entry_price=100.0,
            exit_date=_date_at(bars, exit_idx),
            exit_price=102.5,
            label="t1",
            net_pnl=250.0,
        )
        row = jw.compute_walkaway(bars, position)
        self.assertIsNotNone(row)
        # MFE = 106.5 high - 100 entry; heat before peak = 100 - 98.9 low.
        self.assertAlmostEqual(row["mfe_atr"] * row["atr_at_entry"], 6.5, delta=0.2)
        self.assertGreater(row["heat_before_peak_atr"], 0)
        self.assertLess(row["suggested_stop_price"], 99.0)
        self.assertAlmostEqual(row["suggested_tp_price"], 106.5, delta=0.1)
        self.assertAlmostEqual(row["captured_atr"] * row["atr_at_entry"], 2.5, delta=0.1)
        # 10 sessions after exit, price sits ~105 -> ~+2.5 left on the table.
        self.assertGreater(row["walkaway_10d_atr"], 1.0)
        self.assertIn("LEFT", row["verdicts"])
        self.assertLess(row["capture_ratio_pct"], 50)

    def test_short_side_directionality(self):
        closes = [100.0] * 30 + [100.0, 100.5, 99.0, 97.0, 95.0, 94.0, 94.5, 95.0] + [95.0] * 12
        bars = _bars(closes)
        position = jw.WalkawayPosition(
            source="journal",
            symbol="SHRT",
            side="SHORT",
            entry_date=_date_at(bars, 30),
            entry_price=100.0,
            exit_date=_date_at(bars, 33),
            exit_price=97.0,
            label="t2",
        )
        row = jw.compute_walkaway(bars, position)
        self.assertIsNotNone(row)
        self.assertEqual(row["side"], "SHORT")
        # Peak favorable = 100 - 93.5 low; heat = 101.0 high - 100.
        self.assertAlmostEqual(row["mfe_atr"] * row["atr_at_entry"], 6.5, delta=0.2)
        self.assertGreater(row["suggested_stop_price"], 100.5)
        self.assertGreater(row["captured_atr"], 0)

    def test_focus_pick_without_exit_uses_pick_day_close(self):
        bars, entry_idx = self._long_trade_bars()
        position = jw.WalkawayPosition(
            source="focus",
            symbol="PICK",
            side="LONG",
            entry_date=_date_at(bars, entry_idx),
        )
        row = jw.compute_walkaway(bars, position)
        self.assertIsNotNone(row)
        self.assertEqual(row["entry_price"], 100.0)  # pick-day close
        self.assertNotIn("walkaway_10d_atr", row)
        self.assertGreater(row["mfe_atr"], 0)

    def test_unknown_symbol_or_missing_entry_returns_none(self):
        bars, _ = self._long_trade_bars()
        position = jw.WalkawayPosition(
            source="journal", symbol="X", side="LONG", entry_date="2030-01-01", entry_price=1.0
        )
        self.assertIsNone(jw.compute_walkaway(bars, position))
        self.assertIsNone(jw.compute_walkaway(None, position))


class SummaryTests(unittest.TestCase):
    def test_summary_and_report_render(self):
        bars, entry_idx = ComputeWalkawayTests._long_trade_bars(ComputeWalkawayTests())
        exit_date = _date_at(bars, entry_idx + 4)
        rows = []
        for i in range(3):
            row = jw.compute_walkaway(
                bars,
                jw.WalkawayPosition(
                    source="journal",
                    symbol=f"T{i}",
                    side="LONG",
                    entry_date=_date_at(bars, entry_idx),
                    entry_price=100.0,
                    exit_date=exit_date,
                    exit_price=102.5,
                    label=f"t{i}",
                ),
            )
            rows.append(row)
        summary = jw.summarize_walkaway_rows(rows)
        self.assertEqual(summary["count"], 3)
        self.assertGreater(summary["median_walkaway_10d_atr"], 0)
        self.assertEqual(summary["exited_early_rate_pct"], 100.0)
        self.assertIn("winner_heat_p80_atr", summary)

        report = jw.render_walkaway_report(rows, [])
        self.assertIn("JOURNAL TRADES (3 positions)", report)
        self.assertIn("Could you have held?", report)
        self.assertIn("Stop guidance", report)
        self.assertIn("Actionable flags:", report)

    def test_empty_sections_render(self):
        report = jw.render_walkaway_report([], [])
        self.assertIn("no analyzable positions", report)


class WindowTests(unittest.TestCase):
    def test_closed_trades_are_selected_by_exit_date(self):
        reviewed_week = (pd.Timestamp("2026-08-10").date(), pd.Timestamp("2026-08-14").date())
        exited_this_week = jw.WalkawayPosition(
            source="journal", symbol="OLD", side="LONG", entry_date="2026-07-01",
            exit_date="2026-08-12",
        )
        exits_later = jw.WalkawayPosition(
            source="journal", symbol="NEW", side="LONG", entry_date="2026-08-12",
            exit_date="2026-08-20",
        )

        self.assertTrue(jw._within_window(exited_this_week, *reviewed_week))
        self.assertFalse(jw._within_window(exits_later, *reviewed_week))

    def test_open_focus_picks_fall_back_to_entry_date(self):
        position = jw.WalkawayPosition(
            source="focus", symbol="PICK", side="LONG", entry_date="2026-08-12"
        )
        self.assertTrue(
            jw._within_window(
                position, pd.Timestamp("2026-08-10").date(), pd.Timestamp("2026-08-14").date()
            )
        )


if __name__ == "__main__":
    unittest.main()


def test_one_focus_position_per_name_even_when_it_is_on_two_lists(monkeypatch):
    """The pick store carries a row per CATEGORY since 2026-09-01, so a name on
    both the swing and the M5 list snapshots twice. The walkaway replays a
    POSITION, and the trader was in one - counting it twice would weight that
    name double in every aggregate on the page.

    Fail-before-fix: on the un-fixed loader this returns two positions.
    """
    import human_focus_tracking

    rows = [
        {"trade_date": "2026-06-01", "symbol": "AMGN", "side": "LONG", "source": "focus_m5"},
        {"trade_date": "2026-06-01", "symbol": "AMGN", "side": "LONG", "source": "focus_swing_vetted"},
        {"trade_date": "2026-06-01", "symbol": "AMGN", "side": "SHORT", "source": "focus_m5"},
        {"trade_date": "2026-06-02", "symbol": "AMGN", "side": "LONG", "source": "focus_m5"},
    ]
    monkeypatch.setattr(
        human_focus_tracking, "load_human_focus_daily_picks", lambda *a, **k: rows
    )

    positions = jw.load_focus_positions()

    assert [(p.entry_date, p.symbol, p.side) for p in positions] == [
        ("2026-06-01", "AMGN", "LONG"),
        ("2026-06-01", "AMGN", "SHORT"),
        ("2026-06-02", "AMGN", "LONG"),
    ]
