"""Behavior lock for the market-internals context module.

The bot's regime read is SPY-only; these readings add the volatility, credit,
duration, breadth and concentration axes it cannot see. Advisory-only by
design, so the tests pin the arithmetic and the honest-unknown behavior
rather than any gating.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import market_internals as mi


def series(prev_close, *today_closes, start=datetime(2026, 7, 22, 6, 30)):
    """Two sessions of bars: one prior-day close, then today's closes."""
    bars = [{"dt": start - timedelta(days=1), "close": prev_close}]
    for idx, close in enumerate(today_closes):
        bars.append({"dt": start + timedelta(minutes=5 * idx), "close": close})
    return bars


class SessionChangeTests(unittest.TestCase):
    def test_percent_change_uses_prior_session_close(self):
        # Prior close 100, latest close 102 -> +2%, regardless of intraday path.
        self.assertAlmostEqual(mi.session_change_pct(series(100.0, 101.0, 99.0, 102.0)), 2.0)

    def test_needs_a_prior_session(self):
        today_only = [{"dt": datetime(2026, 7, 22, 6, 30), "close": 100.0},
                      {"dt": datetime(2026, 7, 22, 6, 35), "close": 101.0}]
        self.assertIsNone(mi.session_change_pct(today_only))
        self.assertIsNone(mi.session_change_pct([]))
        self.assertIsNone(mi.session_change_pct(None))

    def test_ignores_unusable_bars(self):
        bars = series(100.0, 102.0)
        bars.insert(1, {"dt": None, "close": 999.0})
        bars.insert(1, {"dt": datetime(2026, 7, 22, 6, 30), "close": None})
        self.assertAlmostEqual(mi.session_change_pct(bars), 2.0)

    def test_accepts_attribute_style_bars(self):
        class Bar:
            def __init__(self, dt, close):
                self.dt, self.close = dt, close

        bars = [Bar(datetime(2026, 7, 21, 12, 55), 50.0), Bar(datetime(2026, 7, 22, 6, 30), 51.0)]
        self.assertAlmostEqual(mi.session_change_pct(bars), 2.0)


class SnapshotTests(unittest.TestCase):
    def _bars(self, **moves):
        """Build series from percent moves off a 100 base."""
        return {sym: series(100.0, 100.0 * (1 + pct / 100.0)) for sym, pct in moves.items()}

    def test_risk_off_tape(self):
        # Volatility bid, credit soft, treasuries bid, breadth lagging.
        snap = mi.build_internals_snapshot(
            self._bars(VXX=3.0, HYG=-0.5, TLT=0.8, RSP=-0.4, MAGS=0.6, SPY=0.0)
        )
        self.assertEqual(snap["readings"]["vol"]["state"], "risk_off")
        self.assertEqual(snap["readings"]["credit"]["state"], "risk_off")
        self.assertEqual(snap["readings"]["duration"]["state"], "risk_off")
        self.assertEqual(snap["readings"]["breadth"]["state"], "risk_off")
        self.assertEqual(snap["tape"], "risk_off_tilt")  # MAGS leading = risk_on
        self.assertTrue(snap["advisory_only"])

    def test_breadth_is_measured_against_spy_not_absolute(self):
        # RSP up 0.5% looks bullish alone, but SPY up 1.5% means the average
        # stock is LAGGING - the index is being carried. That distinction is
        # the entire point of the ratio.
        snap = mi.build_internals_snapshot(self._bars(RSP=0.5, SPY=1.5))
        breadth = snap["readings"]["breadth"]
        self.assertAlmostEqual(breadth["change_pct"], 0.5)
        self.assertAlmostEqual(breadth["spread_vs_spy"], -1.0)
        self.assertEqual(breadth["state"], "risk_off")

    def test_broad_participation_reads_risk_on(self):
        snap = mi.build_internals_snapshot(self._bars(RSP=1.2, SPY=0.4))
        self.assertEqual(snap["readings"]["breadth"]["state"], "risk_on")
        self.assertAlmostEqual(snap["readings"]["breadth"]["spread_vs_spy"], 0.8)

    def test_small_moves_are_flat_not_signal(self):
        snap = mi.build_internals_snapshot(self._bars(VXX=0.05, HYG=-0.02, SPY=0.0))
        self.assertEqual(snap["readings"]["vol"]["state"], "flat")
        self.assertEqual(snap["readings"]["credit"]["state"], "flat")

    def test_missing_symbols_degrade_to_unknown(self):
        snap = mi.build_internals_snapshot(self._bars(VXX=2.0))
        self.assertEqual(snap["readings"]["vol"]["state"], "risk_off")
        self.assertEqual(snap["readings"]["credit"]["state"], "unknown")
        self.assertIsNone(snap["readings"]["breadth"]["spread_vs_spy"])
        self.assertEqual(snap["tape"], "risk_off")

    def test_empty_input_is_unknown_not_a_fabricated_reading(self):
        snap = mi.build_internals_snapshot({})
        self.assertEqual(snap["tape"], "unknown")
        self.assertIsNone(snap["benchmark_change_pct"])
        self.assertEqual(mi.internals_context_fields(snap)["internals_tape"], "unknown")

    def test_mixed_tape_when_axes_disagree(self):
        snap = mi.build_internals_snapshot(self._bars(VXX=2.0, HYG=1.0, SPY=0.0))
        self.assertEqual(snap["tape"], "mixed")


class ContextFieldTests(unittest.TestCase):
    def test_flat_columns_for_the_learning_rows(self):
        snap = mi.build_internals_snapshot(
            {sym: series(100.0, 100.0 * (1 + pct / 100.0))
             for sym, pct in {"VXX": 2.5, "RSP": 0.2, "SPY": 1.0}.items()}
        )
        fields = mi.internals_context_fields(snap)
        self.assertEqual(fields["internals_vol_pct"], 2.5)
        self.assertAlmostEqual(fields["internals_breadth_spread"], -0.8)
        # Volatility bid and the average stock lagging, nothing risk-on to
        # offset them (credit/duration/concentration have no data here).
        self.assertEqual(fields["internals_tape"], "risk_off")

    def test_blank_rather_than_none_for_csv_columns(self):
        fields = mi.internals_context_fields({})
        self.assertEqual(fields["internals_vol_pct"], "")
        self.assertEqual(fields["internals_breadth_spread"], "")


class FormatTests(unittest.TestCase):
    def test_line_names_instruments_and_marks_ratios(self):
        snap = mi.build_internals_snapshot(
            {sym: series(100.0, 100.0 * (1 + pct / 100.0))
             for sym, pct in {"VXX": 2.5, "RSP": 0.2, "SPY": 1.0}.items()}
        )
        line = mi.format_internals_line(snap)
        self.assertIn("VXX +2.50%", line)
        self.assertIn("RSP -0.80% vs SPY", line)
        self.assertIn("risk off", line)

    def test_unavailable_is_explicit(self):
        self.assertEqual(mi.format_internals_line({}), "Internals: unavailable")
        self.assertEqual(mi.format_internals_line(None), "Internals: unavailable")


if __name__ == "__main__":
    unittest.main()
