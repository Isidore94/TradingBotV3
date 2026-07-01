"""Tests for the self-sufficient universe builder (pure parsing/screening only)."""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import universe_builder as ub  # noqa: E402

NASDAQ_SAMPLE = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust|G|N|N|100|Y|N
ZTEST|Test Listing|Q|Y|N|100|N|N
File Creation Time: 0630202522:01|||||||
"""

OTHER_SAMPLE = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
BRK.B|Berkshire Hathaway Class B|N|BRK B|N|100|N|BRK=B
SPY|SPDR S&P 500|P|SPY|Y|100|N|SPY
BAD$|Structured Product|N|BAD$|N|100|N|BAD$
"""


class SymbolDirectoryTests(unittest.TestCase):
    def test_parse_drops_etfs_tests_and_structured(self):
        symbols = ub.parse_symbol_directory(NASDAQ_SAMPLE, OTHER_SAMPLE)
        self.assertIn("AAPL", symbols)
        self.assertIn("BRK-B", symbols)  # dot converted to Yahoo dash form
        self.assertNotIn("QQQ", symbols)  # ETF
        self.assertNotIn("SPY", symbols)  # ETF
        self.assertNotIn("ZTEST", symbols)  # test issue
        self.assertFalse(any("$" in s for s in symbols))


class WeeklysParseTests(unittest.TestCase):
    def test_parse_weeklys_extracts_tickers(self):
        text = 'Available Weeklys,Name\nAAPL,"Apple Inc"\nTSLA,"Tesla"\n"Standard Weeklys",\n'
        symbols = ub.parse_weeklys_csv(text)
        self.assertIn("AAPL", symbols)
        self.assertIn("TSLA", symbols)
        self.assertNotIn("STANDARD WEEKLYS", symbols)


def _history(symbol: str, *, price: float, volume: float, rising: bool, periods: int = 220) -> pd.DataFrame:
    dates = pd.bdate_range("2025-08-01", periods=periods)
    step = 0.2 if rising else -0.2
    start = price - step * periods
    rows = [
        {"symbol": symbol, "datetime": dt, "close": start + step * i, "volume": volume}
        for i, dt in enumerate(dates)
    ]
    return pd.DataFrame(rows)


class ScreenTests(unittest.TestCase):
    def test_metrics_and_screen(self):
        history = pd.concat(
            [
                _history("GOOD", price=50.0, volume=2_000_000, rising=True),
                _history("THIN", price=50.0, volume=100_000, rising=True),
                _history("CHEAP", price=2.0, volume=5_000_000, rising=True),
                _history("DOWN", price=40.0, volume=3_000_000, rising=False),
            ],
            ignore_index=True,
        )
        metrics = ub.compute_universe_metrics(history)
        self.assertEqual(len(metrics), 4)

        screened = ub.apply_universe_screen(
            metrics,
            market_caps_m={"GOOD": 5000.0, "DOWN": 8000.0, "THIN": 5000.0, "CHEAP": 5000.0},
        )
        symbols = set(screened["symbol"])
        self.assertIn("GOOD", symbols)
        self.assertIn("DOWN", symbols)  # base screen keeps downtrends; trend split happens later
        self.assertNotIn("THIN", symbols)
        self.assertNotIn("CHEAP", symbols)

        good = screened[screened["symbol"] == "GOOD"].iloc[0]
        self.assertTrue(good["above_sma_100"] and good["above_sma_200"])
        down = screened[screened["symbol"] == "DOWN"].iloc[0]
        self.assertFalse(down["above_sma_100"] or down["above_sma_200"])

    def test_small_cap_dropped_but_unknown_cap_kept(self):
        metrics = ub.compute_universe_metrics(
            pd.concat(
                [
                    _history("SMALL", price=30.0, volume=2_000_000, rising=True),
                    _history("UNKNOWN", price=30.0, volume=2_000_000, rising=True),
                ],
                ignore_index=True,
            )
        )
        screened = ub.apply_universe_screen(metrics, market_caps_m={"SMALL": 300.0, "UNKNOWN": 0.0})
        symbols = set(screened["symbol"])
        self.assertNotIn("SMALL", symbols)
        self.assertIn("UNKNOWN", symbols)


if __name__ == "__main__":
    unittest.main()
