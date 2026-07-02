"""Tests for the sector/industry RS index board (pure computation, no network)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import industry_scanner as scanner  # noqa: E402


def _frame(daily_return_pct: float, *, periods: int = 90, last_volume_multiple: float = 1.0) -> pd.DataFrame:
    dates = pd.bdate_range("2026-02-02", periods=periods)
    price = 100.0
    rows = []
    for i, dt in enumerate(dates):
        price *= 1.0 + daily_return_pct / 100.0
        volume = 1_000_000 * (last_volume_multiple if i == periods - 1 else 1.0)
        rows.append(
            {"datetime": dt, "open": price, "high": price * 1.01, "low": price * 0.99, "close": price, "volume": volume}
        )
    return pd.DataFrame(rows)


class MetricsTests(unittest.TestCase):
    def test_symbol_metrics_returns_and_volume_buzz(self):
        metrics = scanner.compute_symbol_metrics(_frame(0.5, last_volume_multiple=2.0))
        self.assertAlmostEqual(metrics["pct_change_1d"], 0.5, places=3)
        self.assertAlmostEqual(metrics["return_5d_pct"], (1.005**5 - 1) * 100, places=2)
        self.assertAlmostEqual(metrics["volume_buzz_pct"], 100.0, places=1)

    def test_rs_score_positive_when_beating_benchmark(self):
        strong = scanner.compute_symbol_metrics(_frame(0.8))
        bench = scanner.compute_symbol_metrics(_frame(0.1))
        weak = scanner.compute_symbol_metrics(_frame(-0.4))
        self.assertGreater(scanner.compute_rs_score(strong, bench), 0)
        self.assertLess(scanner.compute_rs_score(weak, bench), 0)
        self.assertIsNone(scanner.compute_rs_score(None, bench))


class BoardTests(unittest.TestCase):
    def test_sector_board_ranks_by_rs(self):
        frames = {
            "SPY": _frame(0.1),
            "XLK": _frame(0.6),
            "XLE": _frame(-0.5),
        }
        rows = scanner.build_sector_board(frames, sector_etfs={"XLK": "Tech", "XLE": "Energy"})
        self.assertEqual([row["etf"] for row in rows], ["XLK", "XLE"])
        self.assertEqual(rows[0]["rs_rank"], 1)
        self.assertGreater(rows[0]["rs_score"], rows[1]["rs_score"])

    def test_industry_board_composites_and_custom_groups(self):
        frames = {
            "SPY": _frame(0.1),
            "AAOI": _frame(0.9, last_volume_multiple=3.0),
            "LITE": _frame(0.7),
            "SLOW1": _frame(-0.2),
            "SLOW2": _frame(-0.3),
        }
        members = {
            "Photonics*": ["AAOI", "LITE"],
            "Slowcoaches": ["SLOW1", "SLOW2"],
            "TooSmall": ["AAOI"],  # below min members -> dropped
        }
        rows = scanner.build_industry_board(frames, members)
        labels = [row["industry"] for row in rows]
        self.assertEqual(labels[0], "Photonics*")
        self.assertNotIn("TooSmall", labels)
        photonics = rows[0]
        self.assertEqual(photonics["member_count"], 2)
        self.assertGreater(photonics["rs_score"], 0)
        self.assertIn("AAOI", photonics["top_movers"])

    def test_collect_industry_members_merges_cache_and_custom(self):
        classifications = {
            "CIEN": {"symbol": "CIEN", "sector": "Technology", "industry": "Communication Equipment"},
            "ANET": {"symbol": "ANET", "sector": "Technology", "industry": "Computer Hardware"},
            "BLANK": {"symbol": "BLANK", "sector": "", "industry": ""},
        }
        members = scanner.collect_industry_members(classifications, {"Photonics": ["AAOI", "LITE"]})
        self.assertEqual(members["Communication Equipment"], ["CIEN"])
        self.assertEqual(members["Photonics*"], ["AAOI", "LITE"])
        self.assertNotIn("", members)

    def test_collect_industry_members_with_index_definitions(self):
        classifications = {
            "NVDA": {"symbol": "NVDA", "sector": "Technology", "industry": "Semiconductors"},
            "AMD": {"symbol": "AMD", "sector": "Technology", "industry": "Semiconductors"},
            "LRCX": {"symbol": "LRCX", "sector": "Technology", "industry": "Semiconductor Equipment & Materials"},
            "ODD": {"symbol": "ODD", "sector": "Consumer", "industry": "Personal Services"},
        }
        definitions = {
            "Semiconductors": {"industries": ["semiconductors"], "tickers": []},  # case-insensitive
            "Chip Equipment": {"industries": ["Semiconductor Equipment & Materials"], "tickers": []},
            "AI Hardware": {"industries": [], "tickers": ["NVDA", "VRT"]},  # overlap + pinned ticker
        }
        members = scanner.collect_industry_members(
            classifications, {"Photonics": ["AAOI"]}, index_definitions=definitions
        )
        self.assertEqual(members["Semiconductors"], ["AMD", "NVDA"])
        self.assertEqual(members["Chip Equipment"], ["LRCX"])
        # NVDA appears in two indexes; pinned tickers join even if unclassified.
        self.assertEqual(members["AI Hardware"], ["NVDA", "VRT"])
        # Unclaimed cache industries fall through as their own rows.
        self.assertEqual(members["Personal Services"], ["ODD"])
        self.assertNotIn("Semiconductor Equipment & Materials", members)
        self.assertEqual(members["Photonics*"], ["AAOI"])

    def test_index_definitions_file_seed_and_parse(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "defs.json"
            definitions = scanner.load_industry_index_definitions(path)
            self.assertIn("Semiconductors", definitions)  # seeded on first run
            self.assertIn("Uranium & Nuclear", definitions)
            payload = {
                "Space": ["rklb", " asts ", ""],  # plain list = tickers shorthand
                "Banks": {"industries": ["Banks - Regional", ""], "tickers": ["jpm"]},
                "": {"tickers": ["IGNORED"]},
                "Empty": {},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            definitions = scanner.load_industry_index_definitions(path)
            self.assertEqual(
                definitions,
                {
                    "Space": {"industries": [], "tickers": ["ASTS", "RKLB"]},
                    "Banks": {"industries": ["Banks - Regional"], "tickers": ["JPM"]},
                },
            )

    def test_custom_groups_file_seed_and_parse(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "groups.json"
            groups = scanner.load_custom_industry_groups(path)
            self.assertIn("Photonics", groups)  # seeded on first run
            path.write_text(json.dumps({"AI Infra": ["nvda", " avgo ", ""]}), encoding="utf-8")
            groups = scanner.load_custom_industry_groups(path)
            self.assertEqual(groups, {"AI Infra": ["AVGO", "NVDA"]})


class RenderTests(unittest.TestCase):
    def test_render_board_text_includes_both_sections(self):
        frames = {"SPY": _frame(0.1), "XLK": _frame(0.6)}
        sector_rows = scanner.build_sector_board(frames, sector_etfs={"XLK": "Tech"})
        industry_rows = scanner.build_industry_board(
            {**frames, "AAOI": _frame(0.9), "LITE": _frame(0.8)}, {"Photonics*": ["AAOI", "LITE"]}
        )
        text = scanner.render_board_text(sector_rows, industry_rows)
        self.assertIn("== SECTORS", text)
        self.assertIn("== INDUSTRY INDEXES", text)
        self.assertIn("XLK", text)
        self.assertIn("Photonics*", text)


if __name__ == "__main__":
    unittest.main()
