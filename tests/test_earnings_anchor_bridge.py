"""The scanner's earnings-anchor caches reach the anchors CSV the warehouse reads.

Before 2026-09-04 the scan computed a current and a previous earnings anchor for
every symbol and kept them only in two JSON caches; `earnings_avwap_anchors.csv`
- the ONLY source `research_warehouse.cli.anchors_from_bronze` reads - held 14
hand-imported rows, so `anchor_instance` had 7 symbols and `swing_house_v1`
graded 0/257 (docs/SWING_SIMULATOR_INVESTIGATION_2026-09-04.md). These tests pin
the bridge: every cached (symbol, date) becomes a CSV row, re-running appends
nothing, the scan calls it right after saving the caches, and a failure never
reaches the scan.
"""

import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib import runner  # noqa: E402
from master_avwap_lib.legacy import EARNINGS_ANCHOR_COLUMNS  # noqa: E402


CURR = {"AAPL": "2026-07-30", "msft": "2026-07-29", "XYZ": "2026-08-05"}
PREV = {"AAPL": "2026-04-30", "MSFT": "2026-04-29", "XYZ": "2026-05-06"}


class BuildCandidatesTests(unittest.TestCase):
    def test_one_candidate_per_symbol_and_anchor_date(self):
        candidates = runner.build_earnings_anchor_bridge_candidates(CURR, PREV)
        keys = sorted((c.ticker, c.anchor_date) for c in candidates)
        self.assertEqual(
            keys,
            [
                ("AAPL", "2026-04-30"),
                ("AAPL", "2026-07-30"),
                ("MSFT", "2026-04-29"),
                ("MSFT", "2026-07-29"),
                ("XYZ", "2026-05-06"),
                ("XYZ", "2026-08-05"),
            ],
        )
        for candidate in candidates:
            self.assertEqual(candidate.source, runner.EARNINGS_ANCHOR_BRIDGE_SOURCE)
            self.assertEqual(candidate.earnings_date, candidate.anchor_date)
            # Nothing the scanner did not compute is invented.
            self.assertEqual(candidate.gap_date, "")
            self.assertEqual(candidate.price, 0.0)
            self.assertEqual(candidate.market_cap, 0)

    def test_side_is_watchlist_membership_not_a_default(self):
        candidates = runner.build_earnings_anchor_bridge_candidates(
            CURR, PREV, longs=["AAPL"], shorts=["XYZ", "AAPL"]
        )
        sides = {c.ticker: c.side for c in candidates}
        self.assertEqual(sides["XYZ"], "SHORT")
        self.assertEqual(sides["AAPL"], "LONG")  # on both lists -> LONG
        self.assertEqual(sides["MSFT"], "LONG")  # on neither -> LONG

    def test_bad_dates_and_blank_symbols_are_skipped_not_coerced(self):
        candidates = runner.build_earnings_anchor_bridge_candidates(
            {"AAPL": "not-a-date", "": "2026-07-30", "MSFT": None, "NVDA": "2026-08-27T00:00:00"},
            {},
        )
        self.assertEqual([(c.ticker, c.anchor_date) for c in candidates], [("NVDA", "2026-08-27")])

    def test_same_date_in_both_caches_yields_one_candidate(self):
        candidates = runner.build_earnings_anchor_bridge_candidates(
            {"AAPL": "2026-07-30"}, {"AAPL": "2026-07-30"}
        )
        self.assertEqual(len(candidates), 1)


class BridgeToCsvTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "earnings_avwap_anchors.csv"

    def test_rows_land_in_the_csv_with_the_anchor_columns(self):
        added = runner.bridge_earnings_anchor_caches_to_csv(CURR, PREV, ["AAPL"], ["XYZ"], path=self.path)
        self.assertEqual(added, 6)
        frame = pd.read_csv(self.path)
        self.assertEqual(list(frame.columns), EARNINGS_ANCHOR_COLUMNS)
        self.assertEqual(len(frame), 6)
        rows = {(r["ticker"], r["anchor_date"]): r for _, r in frame.iterrows()}
        self.assertEqual(rows[("XYZ", "2026-08-05")]["side"], "SHORT")
        self.assertEqual(rows[("AAPL", "2026-07-30")]["side"], "LONG")
        self.assertEqual(rows[("AAPL", "2026-07-30")]["source"], runner.EARNINGS_ANCHOR_BRIDGE_SOURCE)
        self.assertTrue(all(str(v).strip() for v in frame["created_at"]))

    def test_rerun_appends_nothing_and_a_new_anchor_appends_at_the_end(self):
        runner.bridge_earnings_anchor_caches_to_csv(CURR, PREV, path=self.path)
        before = self.path.read_text(encoding="utf-8")
        self.assertEqual(runner.bridge_earnings_anchor_caches_to_csv(CURR, PREV, path=self.path), 0)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

        # The next quarter: AAPL's current anchor rolls; the previous rows stay.
        added = runner.bridge_earnings_anchor_caches_to_csv(
            {**CURR, "AAPL": "2026-10-29"}, {**PREV, "AAPL": "2026-07-30"}, path=self.path
        )
        self.assertEqual(added, 1)
        frame = pd.read_csv(self.path)
        self.assertEqual(len(frame), 7)
        # Appended at the END so bronze's line-offset watermark sees only the new row.
        self.assertEqual((frame.iloc[-1]["ticker"], frame.iloc[-1]["anchor_date"]), ("AAPL", "2026-10-29"))
        self.assertEqual(self.path.read_text(encoding="utf-8")[: len(before)], before)

    def test_hand_imported_rows_survive_the_bridge(self):
        self.path.write_text(
            ",".join(EARNINGS_ANCHOR_COLUMNS)
            + "\nDELL,LONG,2026-02-26,2026-02-27,2026-02-26,amc_inferred,2.209,148.08,9421780,0,Bulk import,bulk_import,2026-03-12T15:46:22\n",
            encoding="utf-8",
        )
        added = runner.bridge_earnings_anchor_caches_to_csv({"DELL": "2026-02-26", "AAPL": "2026-07-30"}, {}, path=self.path)
        self.assertEqual(added, 1)  # DELL's row already exists; it is not duplicated
        frame = pd.read_csv(self.path)
        dell = frame[frame["ticker"] == "DELL"].iloc[0]
        self.assertEqual(dell["source"], "bulk_import")
        self.assertAlmostEqual(float(dell["gap_atr_multiple"]), 2.209)

    def test_a_failure_is_logged_and_never_raised(self):
        with patch.object(runner, "append_anchor_candidates", side_effect=OSError("disk")):
            with self.assertLogs(level="ERROR") as logs:
                self.assertEqual(runner.bridge_earnings_anchor_caches_to_csv(CURR, PREV, path=self.path), 0)
        self.assertTrue(any("Earnings-anchor bridge failed" in line for line in logs.output))

    def test_empty_caches_write_nothing(self):
        self.assertEqual(runner.bridge_earnings_anchor_caches_to_csv({}, {}, path=self.path), 0)
        self.assertFalse(self.path.exists())


class ScanCallSiteTests(unittest.TestCase):
    """The scan calls the bridge once, right after it saves the two caches."""

    def test_bridge_is_called_after_the_cache_saves(self):
        source = inspect.getsource(runner._run_master_impl)
        curr_save = source.index("save_json(CURRENT_CACHE_FILE, curr_cache)")
        prev_save = source.index("save_json(PREV_CACHE_FILE, prev_cache)")
        call = source.index("bridge_earnings_anchor_caches_to_csv(curr_cache, prev_cache, longs, shorts)")
        self.assertLess(curr_save, prev_save)
        self.assertLess(prev_save, call)
        self.assertEqual(source.count("bridge_earnings_anchor_caches_to_csv("), 1)

    def test_the_warehouse_reader_joins_on_the_columns_the_bridge_writes(self):
        # anchors_from_bronze reads `ticker` and `anchor_date` only; the bridge must keep both.
        from research_warehouse import cli

        source = inspect.getsource(cli.anchors_from_bronze)
        self.assertIn('"ticker"', source)
        self.assertIn('"anchor_date"', source)
        candidate = runner.build_earnings_anchor_bridge_candidates({"AAPL": "2026-07-30"}, {})[0]
        self.assertEqual(candidate.ticker, "AAPL")
        self.assertEqual(candidate.anchor_date, "2026-07-30")


if __name__ == "__main__":
    unittest.main()
