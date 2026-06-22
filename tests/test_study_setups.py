"""Tests for the study namespace (B4 / Ticket T6).

A separate tracker namespace (`study_setups`) where new setup ideas are measured
for hit-rate / realized R BEFORE they are allowed to affect scoring. Mirrors the
control/holdout system and MUST stay isolated from Expected-R / calibration /
live ranking — same guarantee as `control_setups`.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402


def _closed_study_setup(symbol, kind, closed_r, *, anchor="2026-01-02", scan="2026-01-03", side="LONG"):
    return {
        "symbol": symbol,
        "side": side,
        "anchor_date": anchor,
        "scan_date": scan,
        "setup_family": kind,
        "study_kind": kind,
        "is_study": True,
        "scenarios": {
            "s1": {
                "tradeable": True,
                "status": "TARGET_HIT" if closed_r > 0 else "STOPPED",
                "total_r": closed_r,
                "stop_reference_label": "LOWER_1",  # primary stop for LONG
            }
        },
    }


def _closed_control_setup(symbol, family, closed_r, reason, *, anchor="2026-01-02", scan="2026-01-03", side="LONG"):
    return {
        "symbol": symbol,
        "side": side,
        "anchor_date": anchor,
        "scan_date": scan,
        "setup_family": family,
        "is_control": True,
        "control_reason": reason,
        "scenarios": {
            "s1": {
                "tradeable": True,
                "status": "TARGET_HIT" if closed_r > 0 else "STOPPED",
                "total_r": closed_r,
                "stop_reference_label": "LOWER_1",
            }
        },
    }


class StudyDiscoveryTests(unittest.TestCase):
    def test_ranks_study_kinds_by_realized_edge_and_flags_promising(self):
        study = {}
        for i in range(6):
            study[f"study:w{i}"] = _closed_study_setup(
                f"W{i}", "hv_level_break", 1.1, anchor=f"2026-01-{i+1:02d}"
            )
        for i in range(6):
            study[f"study:l{i}"] = _closed_study_setup(
                f"L{i}", "compression_break", -0.6, anchor=f"2026-02-{i+1:02d}"
            )
        discovery = m.build_study_discovery_rows({"study_setups": study})
        families = discovery["families"]
        self.assertGreaterEqual(len(families), 2)
        top = families[0]
        self.assertEqual(top["setup_family"], "hv_level_break")
        self.assertAlmostEqual(top["avg_closed_r"], 1.1, places=2)
        self.assertTrue(top["promising"])
        worst = families[-1]
        self.assertLess(worst["avg_closed_r"], 0)
        self.assertFalse(worst["promising"])

    def test_episode_dedupe_counts_one_per_episode(self):
        study = {
            "study:a": _closed_study_setup("NVDA", "hv_level_break", 1.0, anchor="2026-01-02", scan="2026-01-03"),
            "study:b": _closed_study_setup("NVDA", "hv_level_break", 1.0, anchor="2026-01-02", scan="2026-01-06"),
        }
        discovery = m.build_study_discovery_rows({"study_setups": study})
        total = sum(row["closed_episodes"] for row in discovery["families"])
        self.assertEqual(total, 1)

    def test_report_writes_file(self):
        study = {
            f"study:w{i}": _closed_study_setup(f"W{i}", "hv_level_break", 1.1, anchor=f"2026-01-{i+1:02d}")
            for i in range(6)
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "study.txt"
            m.write_master_avwap_study_report(path, {"study_setups": study})
            text = path.read_text(encoding="utf-8")
        self.assertIn("study namespace", text)
        self.assertIn("PROMISING", text)


class StudyIsolationTests(unittest.TestCase):
    def test_study_setups_do_not_leak_into_control_discovery(self):
        promoted = {"p1": _closed_control_setup("AAA", "post_earnings_52w_break", 1.5, "promoted")}
        promoted["p1"].pop("is_control", None)
        promoted["p1"].pop("control_reason", None)
        control = {"control:c1": _closed_control_setup("BBB", "post_earnings_52w_break", -0.5, "near_miss")}

        baseline = m.build_control_discovery_rows({"setups": promoted, "control_setups": control})

        # Add a pile of study setups to the same tracker; control discovery must not change.
        study = {
            f"study:s{i}": _closed_study_setup(f"S{i}", "hv_level_break", 2.0, anchor=f"2026-03-{i+1:02d}")
            for i in range(8)
        }
        with_study = m.build_control_discovery_rows(
            {"setups": promoted, "control_setups": control, "study_setups": study}
        )
        self.assertEqual(baseline, with_study)

    def test_study_discovery_ignores_setups_and_control(self):
        # Only study_setups should feed study discovery.
        discovery = m.build_study_discovery_rows(
            {
                "setups": {"p": _closed_control_setup("X", "foo", 1.0, "promoted")},
                "control_setups": {"control:c": _closed_control_setup("Y", "bar", 1.0, "random")},
                "study_setups": {},
            }
        )
        self.assertEqual(discovery["families"], [])


class StudyPayloadRoundTripTests(unittest.TestCase):
    def test_default_payload_has_study_namespace(self):
        self.assertIn("study_setups", m._default_setup_tracker_payload())

    def test_save_and_load_preserves_study_setups(self):
        study = {"study:a": _closed_study_setup("NVDA", "hv_level_break", 1.0)}
        with tempfile.TemporaryDirectory() as td:
            tracker_path = Path(td) / "tracker.json"
            with patch.object(m, "SETUP_TRACKER_FILE", tracker_path):
                payload = m._default_setup_tracker_payload()
                payload["study_setups"] = study
                m.save_setup_tracker_payload(payload)
                reloaded = m.load_setup_tracker_payload()
        self.assertIn("study:a", reloaded["study_setups"])
        self.assertEqual(reloaded["study_setups"]["study:a"]["study_kind"], "hv_level_break")


class PruneStudySetupsTests(unittest.TestCase):
    def test_drops_records_older_than_keep_days(self):
        study = {
            "old": {"symbol": "A", "scan_date": "2024-01-01"},
            "fresh": {"symbol": "B", "scan_date": "2026-06-10"},
        }
        m._prune_study_setups(study, reference_scan_date="2026-06-21")
        self.assertNotIn("old", study)
        self.assertIn("fresh", study)

    def test_caps_total_record_count(self):
        study = {
            f"s{i}": {"symbol": f"S{i}", "scan_date": f"2026-06-{(i % 28) + 1:02d}"}
            for i in range(m.STUDY_SETUP_MAX_RECORDS + 25)
        }
        m._prune_study_setups(study, reference_scan_date="2026-06-21")
        self.assertLessEqual(len(study), m.STUDY_SETUP_MAX_RECORDS)


if __name__ == "__main__":
    unittest.main()
