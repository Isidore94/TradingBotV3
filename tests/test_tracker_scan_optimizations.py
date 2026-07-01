"""Tracker recompute scan-cost optimizations (fixes B / E / D).

B: a CLOSED setup whose forward window ended past the daily-bar restatement buffer
   is "sealed" -- replaying it reproduces the identical record, so it is skipped.
E: symbols whose every tracked setup is closed/untradeable no longer need a live
   daily-bar refetch during recompute (`_tracker_setup_needs_live_bars`).
D: the large tracker payload is loaded once and threaded through, so
   `update_setup_tracker_from_scan` reuses a passed payload instead of reloading.
"""

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402


def _closed(last_mark: str) -> dict:
    return {"symbol": "X", "setup_status": "CLOSED", "latest_snapshot": {"trade_date": last_mark}}


class SealedSetupPredicateTests(unittest.TestCase):
    SCAN = "2026-07-01"

    def test_closed_and_aged_is_sealed(self):
        self.assertTrue(m._tracker_setup_recompute_is_sealed(_closed("2026-05-01"), self.SCAN))

    def test_closed_but_recent_is_not_sealed(self):
        self.assertFalse(m._tracker_setup_recompute_is_sealed(_closed("2026-06-25"), self.SCAN))

    def test_open_is_never_sealed(self):
        setup = _closed("2026-05-01")
        setup["setup_status"] = "OPEN"
        self.assertFalse(m._tracker_setup_recompute_is_sealed(setup, self.SCAN))

    def test_missing_mark_is_not_sealed(self):
        self.assertFalse(m._tracker_setup_recompute_is_sealed({"setup_status": "CLOSED"}, self.SCAN))

    def test_threshold_boundary(self):
        scan = date(2026, 7, 1)
        at_threshold = _closed((scan - timedelta(days=21)).isoformat())
        past_threshold = _closed((scan - timedelta(days=22)).isoformat())
        self.assertFalse(m._tracker_setup_recompute_is_sealed(at_threshold, scan.isoformat()))
        self.assertTrue(m._tracker_setup_recompute_is_sealed(past_threshold, scan.isoformat()))

    def test_last_mark_prefers_snapshot_then_daily_marks(self):
        self.assertEqual(m._tracker_setup_last_mark_date(_closed("2026-05-02")), "2026-05-02")
        via_marks = {
            "setup_status": "CLOSED",
            "daily_marks": [{"trade_date": "2026-05-01"}, {"trade_date": "2026-05-02"}],
        }
        self.assertEqual(m._tracker_setup_last_mark_date(via_marks), "2026-05-02")


class NeedsLiveBarsTests(unittest.TestCase):
    def test_open_needs_live(self):
        self.assertTrue(m._tracker_setup_needs_live_bars({"setup_status": "OPEN"}))

    def test_closed_and_untradeable_do_not(self):
        self.assertFalse(m._tracker_setup_needs_live_bars({"setup_status": "CLOSED"}))
        self.assertFalse(m._tracker_setup_needs_live_bars({"setup_status": "UNTRADEABLE"}))

    def test_missing_status_defaults_to_live(self):
        self.assertTrue(m._tracker_setup_needs_live_bars({}))


class PayloadThreadingTests(unittest.TestCase):
    def test_update_reuses_passed_payload_without_reloading(self):
        payload = {"setups": {}, "control_setups": {}, "study_setups": {}, "daily_watchlists": {}}
        with mock.patch.object(m, "load_setup_tracker_payload", side_effect=AssertionError("must not reload")), \
             mock.patch.object(m, "export_setup_tracker_views"), \
             mock.patch.object(m, "write_control_discovery_report"), \
             mock.patch.object(m, "write_master_avwap_study_report"), \
             mock.patch.object(m, "save_setup_tracker_payload") as save_mock:
            m.update_setup_tracker_from_scan(
                [],
                {"symbols": {}},
                {},
                {},
                None,
                auto_tune=False,
                tracker_payload=payload,
            )
        # Saved the very object we passed in -> it was reused, not reloaded from disk.
        save_mock.assert_called_once()
        self.assertIs(save_mock.call_args.args[0], payload)


if __name__ == "__main__":
    unittest.main()
