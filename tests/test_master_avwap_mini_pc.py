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
    def test_watchlist_filter_scan_includes_theta_and_shared_swing_paths(self):
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
        self.assertTrue(kwargs["include_theta"])

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
