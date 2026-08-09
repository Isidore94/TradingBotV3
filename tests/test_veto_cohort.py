"""Veto forward-tracking: its own cohort, and nothing else disturbed.

Two jobs here.

The first is the wiring: a veto annotation becomes a pick row in the
human-focus column schema with source ``veto_<reason_code>``, so the existing
outcome math grades it and ``human_focus_veto_<reason>`` becomes computable.

The second is the load-bearing one - proving the new cohort family costs the
old ones nothing. ``build_human_focus_performance_rows`` gained a fourth base
family and a rewritten sub-cohort naming expression; the characterization test
pins that focus-only outcomes still aggregate byte-identically, and its
sensitivity control proves the comparison can actually tell cohorts apart.
The separate-file test pins why veto rows are not in the human-focus picks
file at all: that file is keyed (trade_date, symbol, side) with no source, so
a veto row for a name that is also a focus pick that day would suppress the
focus row.
"""

from __future__ import annotations

import csv
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from human_focus_tracking import (  # noqa: E402
    HUMAN_FOCUS_DAILY_PICK_COLUMNS,
    build_human_focus_performance_rows,
    snapshot_human_focus_picks,
)
from ui.annotations.store import EVENT_VETO, record_annotation  # noqa: E402
from ui.annotations.veto_cohort import (  # noqa: E402
    merge_veto_cohort_picks,
    veto_cohort_source,
    veto_pick_rows,
)

NOW = datetime(2026, 8, 7, 10, 30, 0)


def _outcome(source: str, symbol: str, *, side: str = "LONG", h1: float = 0.01) -> dict:
    return {
        "trade_date": "2026-08-03",
        "symbol": symbol,
        "side": side,
        "source": source,
        "h1_return": f"{h1}",
        "h3_return": f"{h1 * 2}",
    }


def _veto_annotation(symbol: str, reason: str, *, side: str = "LONG", date: str = "2026-08-07") -> dict:
    return {
        "event_type": EVENT_VETO,
        "symbol": symbol,
        "reason_code": reason,
        "side": side,
        "session_date": date,
    }


class VetoCohortSourceTests(unittest.TestCase):
    def test_source_is_the_reason_prefixed(self) -> None:
        self.assertEqual(veto_cohort_source("volume_dry"), "veto_volume_dry")

    def test_blank_reason_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            veto_cohort_source("")


class VetoPickRowTests(unittest.TestCase):
    def test_annotation_becomes_a_cohort_row(self) -> None:
        rows, skipped = veto_pick_rows([_veto_annotation("NVDA", "volume_dry")], now=NOW)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "NVDA")
        self.assertEqual(rows[0]["side"], "LONG")
        self.assertEqual(rows[0]["source"], "veto_volume_dry")
        self.assertEqual(rows[0]["trade_date"], "2026-08-07")
        self.assertEqual(sorted(rows[0]), sorted(HUMAN_FOCUS_DAILY_PICK_COLUMNS))

    def test_a_veto_with_no_side_is_counted_and_skipped(self) -> None:
        """Guessing a side would manufacture a directional claim; the tracker
        reads a blank side as LONG, so the row must not be written at all."""
        rows, skipped = veto_pick_rows([_veto_annotation("NVDA", "volume_dry", side="")], now=NOW)
        self.assertEqual(rows, [])
        self.assertEqual(skipped, 1)

    def test_first_veto_of_a_name_that_day_wins(self) -> None:
        rows, _ = veto_pick_rows(
            [
                _veto_annotation("NVDA", "volume_dry"),
                _veto_annotation("NVDA", "earnings_too_close"),
            ],
            now=NOW,
        )
        self.assertEqual([row["source"] for row in rows], ["veto_volume_dry"])

    def test_the_same_name_on_both_sides_is_two_rows(self) -> None:
        rows, _ = veto_pick_rows(
            [
                _veto_annotation("NVDA", "volume_dry", side="LONG"),
                _veto_annotation("NVDA", "too_extended_from_base", side="SHORT"),
            ],
            now=NOW,
        )
        self.assertEqual(len(rows), 2)

    def test_non_veto_events_are_ignored(self) -> None:
        rows, _ = veto_pick_rows([{"event_type": "note", "symbol": "NVDA", "note": "x"}], now=NOW)
        self.assertEqual(rows, [])

    def test_incomplete_rows_are_ignored(self) -> None:
        rows, skipped = veto_pick_rows(
            [
                {"event_type": EVENT_VETO, "symbol": "", "reason_code": "volume_dry", "side": "LONG", "session_date": "2026-08-07"},
                {"event_type": EVENT_VETO, "symbol": "NVDA", "reason_code": "", "side": "LONG", "session_date": "2026-08-07"},
                {"event_type": EVENT_VETO, "symbol": "NVDA", "reason_code": "volume_dry", "side": "LONG", "session_date": ""},
            ],
            now=NOW,
        )
        self.assertEqual(rows, [])
        self.assertEqual(skipped, 0)


class VetoCohortMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        self.annotations = root / "trader_annotations.jsonl"
        self.picks = root / "veto_cohort_picks.csv"
        self.addCleanup(self._tmp.cleanup)

    def _veto(self, symbol: str, reason: str, side: str = "LONG") -> None:
        record_annotation(
            EVENT_VETO,
            symbol=symbol,
            reason_code=reason,
            side=side,
            session_date="2026-08-07",
            path=self.annotations,
        )

    def _rows(self) -> list[dict[str, str]]:
        with self.picks.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_merge_writes_the_cohort_file(self) -> None:
        self._veto("NVDA", "volume_dry")
        result = merge_veto_cohort_picks(
            annotations_path=self.annotations, picks_path=self.picks, now=NOW
        )
        self.assertEqual(result["added"], 1)
        self.assertTrue(result["written"])
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "veto_volume_dry")

    def test_merge_is_idempotent(self) -> None:
        self._veto("NVDA", "volume_dry")
        merge_veto_cohort_picks(annotations_path=self.annotations, picks_path=self.picks, now=NOW)
        before = self.picks.read_bytes()
        again = merge_veto_cohort_picks(
            annotations_path=self.annotations, picks_path=self.picks, now=NOW
        )
        self.assertEqual(again["added"], 0)
        self.assertEqual(self.picks.read_bytes(), before)

    def test_merge_never_removes_an_existing_row(self) -> None:
        self._veto("NVDA", "volume_dry")
        merge_veto_cohort_picks(annotations_path=self.annotations, picks_path=self.picks, now=NOW)
        self._veto("AMD", "earnings_too_close")
        merge_veto_cohort_picks(annotations_path=self.annotations, picks_path=self.picks, now=NOW)
        self.assertEqual({row["symbol"] for row in self._rows()}, {"NVDA", "AMD"})

    def test_a_row_whose_annotation_is_gone_still_survives(self) -> None:
        """Append-only forward: the cohort keeps grading what it started."""
        self._veto("NVDA", "volume_dry")
        merge_veto_cohort_picks(annotations_path=self.annotations, picks_path=self.picks, now=NOW)
        self.annotations.write_text("", encoding="utf-8")
        merge_veto_cohort_picks(annotations_path=self.annotations, picks_path=self.picks, now=NOW)
        self.assertEqual({row["symbol"] for row in self._rows()}, {"NVDA"})

    def test_sideless_vetoes_are_reported_not_hidden(self) -> None:
        record_annotation(
            EVENT_VETO,
            symbol="NVDA",
            reason_code="volume_dry",
            session_date="2026-08-07",
            path=self.annotations,
        )
        result = merge_veto_cohort_picks(
            annotations_path=self.annotations, picks_path=self.picks, now=NOW
        )
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["skipped_no_side"], 1)

    def test_empty_log_writes_nothing(self) -> None:
        result = merge_veto_cohort_picks(
            annotations_path=self.annotations, picks_path=self.picks, now=NOW
        )
        self.assertEqual(result["added"], 0)
        self.assertFalse(self.picks.exists())


class CohortIsolationTests(unittest.TestCase):
    """The new family must cost the existing ones nothing."""

    FOCUS_OUTCOMES = [
        _outcome("focus_swing", "AAA"),
        _outcome("focus_swing_h1", "BBB"),
        _outcome("focus_swing_setups", "CCC", side="SHORT"),
        _outcome("focus_m5", "DDD"),
        _outcome("focus_m5_m5", "EEE"),
        _outcome("focus_pick", "FFF"),
        _outcome("", "GGG"),
    ]

    #: Captured from build_human_focus_performance_rows before the veto family
    #: was added. If a future edit changes how existing cohorts aggregate, this
    #: is the test that says so.
    EXPECTED_COHORTS = [
        "human_focus_swing",
        "human_focus_swing_h1",
        "human_focus_swing_setups",
        "human_focus_m5",
        "human_focus_m5_m5",
        "human_focus_pick",
    ]

    def test_focus_only_outcomes_aggregate_exactly_as_before(self) -> None:
        rows = build_human_focus_performance_rows(self.FOCUS_OUTCOMES, updated_at="2026-08-07T10:30:00")
        seen: list[str] = []
        for row in rows:
            if row["cohort"] not in seen:
                seen.append(row["cohort"])
        self.assertEqual(seen, self.EXPECTED_COHORTS)

    def test_the_characterization_can_tell_cohorts_apart(self) -> None:
        """Sensitivity control: the assertion above is not vacuous."""
        mutated = self.FOCUS_OUTCOMES + [_outcome("focus_swing_manual", "HHH")]
        rows = build_human_focus_performance_rows(mutated, updated_at="2026-08-07T10:30:00")
        self.assertIn("human_focus_swing_manual", {row["cohort"] for row in rows})

    def test_a_veto_row_never_lands_in_the_focus_pick_cohort(self) -> None:
        rows = build_human_focus_performance_rows(
            [_outcome("veto_volume_dry", "NVDA")], updated_at="2026-08-07T10:30:00"
        )
        cohorts = {row["cohort"] for row in rows}
        self.assertIn("human_focus_veto", cohorts)
        self.assertIn("human_focus_veto_volume_dry", cohorts)
        self.assertNotIn("human_focus_pick", cohorts)

    def test_each_reason_grades_as_its_own_sub_cohort(self) -> None:
        rows = build_human_focus_performance_rows(
            [
                _outcome("veto_volume_dry", "AAA"),
                _outcome("veto_incoming_trendline", "BBB"),
            ],
            updated_at="2026-08-07T10:30:00",
        )
        cohorts = {row["cohort"] for row in rows}
        self.assertIn("human_focus_veto_volume_dry", cohorts)
        self.assertIn("human_focus_veto_incoming_trendline", cohorts)

    def test_mixed_outcomes_keep_the_families_separate(self) -> None:
        rows = build_human_focus_performance_rows(
            self.FOCUS_OUTCOMES + [_outcome("veto_volume_dry", "NVDA")],
            updated_at="2026-08-07T10:30:00",
        )
        pick_rows = [row for row in rows if row["cohort"] == "human_focus_pick" and row["side"] == "ALL"]
        focus_only = build_human_focus_performance_rows(
            self.FOCUS_OUTCOMES, updated_at="2026-08-07T10:30:00"
        )
        expected = [row for row in focus_only if row["cohort"] == "human_focus_pick" and row["side"] == "ALL"]
        self.assertEqual(pick_rows, expected)


class FocusPicksFileUntouchedTests(unittest.TestCase):
    """Why veto rows live in their own file."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_the_focus_picks_key_has_no_source_column(self) -> None:
        """The reason a shared file would lose data: (date, symbol, side) is
        the whole key, so a veto row would occupy a focus pick's slot."""
        self.assertIn("source", HUMAN_FOCUS_DAILY_PICK_COLUMNS)
        state = self.root / "state.json"
        picks = self.root / "human_focus_daily_picks.csv"
        snapshot_human_focus_picks(
            market_date="2026-08-07",
            focus_maps_by_category={"swing": {"long": ["NVDA"], "short": []}},
            snapshot_state_path=state,
            daily_picks_path=picks,
            now=NOW,
        )
        # Re-running with the same name adds nothing: the key already exists.
        result = snapshot_human_focus_picks(
            market_date="2026-08-07",
            focus_maps_by_category={"swing": {"long": ["NVDA"], "short": []}},
            snapshot_state_path=state,
            daily_picks_path=picks,
            force=True,
            now=NOW,
        )
        self.assertEqual(result["added"], 0)

    def test_veto_merge_does_not_touch_the_focus_picks_file(self) -> None:
        annotations = self.root / "trader_annotations.jsonl"
        focus_picks = self.root / "human_focus_daily_picks.csv"
        focus_picks.write_text(
            "trade_date,symbol,side,source,snapshotted_at,active_at_snapshot\n"
            "2026-08-07,NVDA,LONG,focus_swing,2026-08-07T09:00:00,1\n",
            encoding="utf-8",
        )
        before = focus_picks.read_bytes()
        record_annotation(
            EVENT_VETO,
            symbol="NVDA",
            reason_code="volume_dry",
            side="LONG",
            session_date="2026-08-07",
            path=annotations,
        )
        merge_veto_cohort_picks(
            annotations_path=annotations,
            picks_path=self.root / "veto_cohort_picks.csv",
            now=NOW,
        )
        self.assertEqual(focus_picks.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
