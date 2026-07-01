"""Fix C: sealed-record compaction drops per-bar replay detail while preserving every
scalar outcome the ranking/stats aggregations read.
"""

import copy
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402


def _mark(d):
    return {
        "trade_date": d,
        "close": 100.0,
        "current_levels": {"UPPER_1": 101.0},
        "feature_snapshot": {"atr20": 2.0},
        "scenario_events": [{"reason": "STOP", "shares": 100, "price": 98.0}],
    }


def _record(status, last_mark, *, n_marks=3):
    marks = [_mark((date.fromisoformat(last_mark) - timedelta(days=n_marks - 1 - i)).isoformat()) for i in range(n_marks)]
    return {
        "symbol": "AAA",
        "setup_status": status,
        "latest_snapshot": dict(marks[-1]),
        "entry_feature_snapshot": {"ema8": 50.0},
        "daily_marks": marks,
        "scenarios": {
            "s1": {
                "status": "STOPPED",
                "total_r": -1.0,
                "realized_r": -1.0,
                "tradeable": True,
                "stop_reference_label": "LOWER_1",
                "events": [{"reason": "STOP", "shares": 100, "price": 98.0}],
            }
        },
    }


class CompactRecordTests(unittest.TestCase):
    def test_strips_marks_and_events_keeps_outcomes(self):
        setup = _record("CLOSED", "2026-05-01")
        changed = m._compact_tracker_setup_record(setup)
        self.assertTrue(changed)
        self.assertEqual(setup["daily_marks"], [])
        self.assertEqual(setup["scenarios"]["s1"]["events"], [])
        # Scalar outcomes + snapshots preserved.
        self.assertEqual(setup["scenarios"]["s1"]["total_r"], -1.0)
        self.assertEqual(setup["scenarios"]["s1"]["stop_reference_label"], "LOWER_1")
        self.assertEqual(setup["latest_snapshot"]["trade_date"], "2026-05-01")
        self.assertEqual(setup["entry_feature_snapshot"], {"ema8": 50.0})

    def test_idempotent(self):
        setup = _record("CLOSED", "2026-05-01")
        self.assertTrue(m._compact_tracker_setup_record(setup))
        self.assertFalse(m._compact_tracker_setup_record(setup))

    def test_outcome_summary_unchanged_by_compaction(self):
        setup = _record("CLOSED", "2026-05-01")
        before = m._summarize_tracker_setup_outcome(copy.deepcopy(setup))
        m._compact_tracker_setup_record(setup)
        after = m._summarize_tracker_setup_outcome(setup)
        self.assertEqual(before, after)


class CompactSealedNamespaceTests(unittest.TestCase):
    def test_only_sealed_records_are_compacted(self):
        scan = "2026-07-01"
        tracker = {
            "setups": {
                "sealed": _record("CLOSED", "2026-05-01"),      # old closed -> sealed
                "recent_closed": _record("CLOSED", "2026-06-28"),  # closed < 21d -> not sealed
                "open": _record("OPEN", "2026-05-01"),           # open -> never sealed
            },
            "control_setups": {},
            "study_setups": {},
        }
        n = m._compact_sealed_tracker_setups(tracker, scan)
        self.assertEqual(n, 1)
        self.assertEqual(tracker["setups"]["sealed"]["daily_marks"], [])
        self.assertNotEqual(tracker["setups"]["recent_closed"]["daily_marks"], [])
        self.assertNotEqual(tracker["setups"]["open"]["daily_marks"], [])

    def test_second_pass_is_noop(self):
        scan = "2026-07-01"
        tracker = {"setups": {"sealed": _record("CLOSED", "2026-05-01")}, "control_setups": {}, "study_setups": {}}
        self.assertEqual(m._compact_sealed_tracker_setups(tracker, scan), 1)
        self.assertEqual(m._compact_sealed_tracker_setups(tracker, scan), 0)


if __name__ == "__main__":
    unittest.main()
