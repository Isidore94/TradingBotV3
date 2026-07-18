import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap_mini_pc as mini_pc  # noqa: E402


class MasterAvwapMiniPCTests(unittest.TestCase):
    def test_watchlist_filter_scan_uses_shared_swing_paths(self):
        # Theta is unconditional inside run_master now (deferred enrichment);
        # the mini-PC caller must NOT pass the removed include_theta kwarg —
        # doing so crashed every scheduled scan with a TypeError (2026-07-10).
        filter_summary = {"status": "ok", "message": "checked"}
        with patch.object(mini_pc, "filter_watchlists_by_previous_day_levels", return_value=filter_summary), patch.object(
            mini_pc,
            "run_master",
            return_value={"setup_tracker_updated": True},
        ) as run_master:
            returned_filter, scan_result = mini_pc.run_master_with_watchlist_filter(update_setup_tracker=False)

        self.assertIs(returned_filter, filter_summary)
        self.assertEqual(scan_result, {"setup_tracker_updated": True})
        self.assertEqual(run_master.call_count, 1)
        kwargs = run_master.call_args.kwargs
        self.assertTrue(kwargs["use_shared_watchlists"])
        self.assertFalse(kwargs["update_setup_tracker"])
        self.assertTrue(kwargs["require_ib_for_setup_tracker"])
        self.assertNotIn("include_theta", kwargs)

    def test_eod_journal_import_guard_runs_once_per_date(self):
        state = mini_pc.default_state(["10:00"])
        target = date(2026, 6, 15)

        self.assertTrue(mini_pc.should_run_eod_journal_import(state, target))

        state["journal_eod_import"] = {
            "date": target.isoformat(),
            "status": "success",
        }
        self.assertFalse(mini_pc.should_run_eod_journal_import(state, target))
        self.assertTrue(mini_pc.should_run_eod_journal_import(state, date(2026, 6, 16)))


if __name__ == "__main__":
    unittest.main()


class SetupTrackerPurityQuarantineTests(unittest.TestCase):
    """2026-07-17: 1-2 chronic Yahoo symbols must not veto a whole week of
    tracker writes; a large non-IB fraction (systemic fallback) still does."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd

        from master_avwap_lib import runner
        from master_avwap_lib.legacy import DAILY_BAR_SOURCE_IBKR, _set_daily_bar_source

        cls.runner = runner
        cls.pd = pd
        cls.ib_source = DAILY_BAR_SOURCE_IBKR
        cls.set_source = staticmethod(_set_daily_bar_source)

    def _frames(self, sources_by_symbol):
        return {
            symbol: self.set_source(self.pd.DataFrame({"close": [1.0]}), source)
            for symbol, source in sources_by_symbol.items()
        }

    def test_clean_run_passes_with_no_quarantine(self):
        frames = self._frames({"AAPL": self.ib_source, "MSFT": self.ib_source})
        allowed, quarantined, reason = self.runner.evaluate_setup_tracker_purity(
            ["AAPL", "MSFT"], frames
        )
        self.assertTrue(allowed)
        self.assertEqual(quarantined, [])
        self.assertEqual(reason, "")

    def test_small_dirty_tail_is_quarantined_not_vetoed(self):
        sources = {f"SYM{i}": self.ib_source for i in range(9)}
        sources["LC"] = "yahoo"
        allowed, quarantined, reason = self.runner.evaluate_setup_tracker_purity(
            list(sources), self._frames(sources)
        )
        self.assertTrue(allowed)
        self.assertEqual(quarantined, ["LC"])
        self.assertEqual(reason, "")

    def test_systemic_fallback_still_vetoes(self):
        sources = {"AAPL": "yahoo", "MSFT": "yahoo", "NVDA": self.ib_source}
        allowed, quarantined, reason = self.runner.evaluate_setup_tracker_purity(
            list(sources), self._frames(sources)
        )
        self.assertFalse(allowed)
        self.assertEqual(sorted(quarantined), ["AAPL", "MSFT"])
        self.assertIn("non-IBKR daily data", reason)
        self.assertIn("AAPL", reason)

    def test_missing_frame_counts_as_dirty(self):
        sources = {f"SYM{i}": self.ib_source for i in range(9)}
        frames = self._frames(sources)
        allowed, quarantined, reason = self.runner.evaluate_setup_tracker_purity(
            [*sources, "GHOST"], frames
        )
        self.assertTrue(allowed)
        self.assertEqual(quarantined, ["GHOST"])
