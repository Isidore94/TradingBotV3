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
        # This fixture is a bare dict with no vocab_version, so the key falls
        # back to the unversioned form - which is exactly what keeps rows
        # already on disk grading in the cohort they were filed under. A
        # version-carrying annotation is covered in test_veto_cohort_grading.
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
                {"event_type": EVENT_VETO, "symbol": "NVDA", "reason_code": "volume_dry", "side": "LONG", "session_date": ""},
            ],
            now=NOW,
        )
        self.assertEqual(rows, [])

    def test_a_codeless_veto_is_graded_rather_than_ignored(self):
        """CHANGED BY P10, deliberately.

        A row with no `reason_code` used to be dropped here with the rows that
        have no symbol and no date - which put "Not today", the desk's most-used
        dismissal, outside every forward record. It now grades under
        `veto_uncoded`, its own cohort, never pooled with a coded one.
        """
        rows, skipped = veto_pick_rows(
            [
                {"event_type": EVENT_VETO, "symbol": "NVDA", "reason_code": "", "side": "LONG", "session_date": "2026-08-07"},
            ],
            now=NOW,
        )
        self.assertEqual([row["source"] for row in rows], ["veto_uncoded"])
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
        # record_annotation stamps vocab_version, so a row written through the
        # real writer carries the versioned key (2026-08-20). Read the version
        # from the vocabulary, never assert a literal one.
        from ui.annotations.vocabulary import load_veto_vocabulary

        version = load_veto_vocabulary().vocab_version
        self.assertEqual(rows[0]["source"], f"veto_v{version}_volume_dry")

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


def _characterization_outcome(source, symbol, side="LONG", h1=None, h3=None, h5=None, h10=None):
    row = {"trade_date": "2026-08-03", "symbol": symbol, "side": side, "source": source}
    for name, value in (("h1", h1), ("h3", h3), ("h5", h5), ("h10", h10)):
        row[f"{name}_return"] = "" if value is None else f"{value}"
    return row


#: Focus-only outcomes spanning every base family, sub-cohorts, both sides,
#: all four horizons, missing returns, an unparseable return, an untagged
#: legacy row, and profit-factor edge cases (all-gain -> "inf", all-loss,
#: mixed). Rich enough that a change to ordering, counting, win-rate, return
#: averaging, profit-factor, or side/horizon filtering shows up in the golden.
CHARACTERIZATION_OUTCOMES = [
    _characterization_outcome("focus_swing", "AAA", h1=0.01, h3=0.02, h5=-0.01),
    _characterization_outcome("focus_swing", "AAB", side="SHORT", h1=-0.005, h3=0.0, h10=0.04),
    _characterization_outcome("focus_swing_h1", "BBB", h1=0.03),
    _characterization_outcome("focus_swing_setups", "CCC", side="SHORT", h1=0.002, h3=-0.002),
    _characterization_outcome("focus_swing_manual", "CCD", h1=-0.04, h3=-0.08),
    _characterization_outcome("focus_m5", "DDD", h1=0.0, h3=0.001),
    _characterization_outcome("focus_m5_m5", "EEE", h1=0.015),
    _characterization_outcome("focus_m5_m5", "EEF", h1=-0.015),
    _characterization_outcome("focus_pick", "FFF", h1=0.02, h3=0.01, h5=0.005, h10=-0.03),
    _characterization_outcome("", "GGG", h1=0.007),
    _characterization_outcome("unknown_origin", "HHH", h1=-0.001),
    _characterization_outcome("focus_pick", "III", h1="bad-not-a-float"),
]

_TS = "2026-08-07T10:30:00"


def _golden(cohort, side, horizon, samples, win_rate, avg, pf):
    return {
        "cohort": cohort,
        "side": side,
        "horizon_sessions": horizon,
        "sample_count": samples,
        "win_rate": win_rate,
        "avg_side_return": avg,
        "profit_factor": pf,
        "updated_at": _TS,
    }


#: Generated by running origin/main's PRE-CHANGE build_human_focus_performance_rows
#: (before the veto family and the COHORT_BASE_BY_SOURCE_PREFIX rewrite) over
#: CHARACTERIZATION_OUTCOMES, and verified identical against the rewritten
#: implementation at the time the fence was strengthened. Every field of every
#: row, in order. Do not regenerate this from current code to make a red test
#: green - a mismatch here IS the finding.
CHARACTERIZATION_GOLDEN = [
    _golden("human_focus_swing", "ALL", "1", "5", "0.6000", "-0.000600", "0.9333"),
    _golden("human_focus_swing", "ALL", "3", "4", "0.2500", "-0.015500", "0.2439"),
    _golden("human_focus_swing", "ALL", "5", "1", "0.0000", "-0.010000", "0.0000"),
    _golden("human_focus_swing", "ALL", "10", "1", "1.0000", "0.040000", "inf"),
    _golden("human_focus_swing", "LONG", "1", "3", "0.6667", "-0.000000", "1.0000"),
    _golden("human_focus_swing", "LONG", "3", "2", "0.5000", "-0.030000", "0.2500"),
    _golden("human_focus_swing", "LONG", "5", "1", "0.0000", "-0.010000", "0.0000"),
    _golden("human_focus_swing", "SHORT", "1", "2", "0.5000", "-0.001500", "0.4000"),
    _golden("human_focus_swing", "SHORT", "3", "2", "0.0000", "-0.001000", "0.0000"),
    _golden("human_focus_swing", "SHORT", "10", "1", "1.0000", "0.040000", "inf"),
    _golden("human_focus_swing_h1", "ALL", "1", "1", "1.0000", "0.030000", "inf"),
    _golden("human_focus_swing_h1", "LONG", "1", "1", "1.0000", "0.030000", "inf"),
    _golden("human_focus_swing_manual", "ALL", "1", "1", "0.0000", "-0.040000", "0.0000"),
    _golden("human_focus_swing_manual", "ALL", "3", "1", "0.0000", "-0.080000", "0.0000"),
    _golden("human_focus_swing_manual", "LONG", "1", "1", "0.0000", "-0.040000", "0.0000"),
    _golden("human_focus_swing_manual", "LONG", "3", "1", "0.0000", "-0.080000", "0.0000"),
    _golden("human_focus_swing_setups", "ALL", "1", "1", "1.0000", "0.002000", "inf"),
    _golden("human_focus_swing_setups", "ALL", "3", "1", "0.0000", "-0.002000", "0.0000"),
    _golden("human_focus_swing_setups", "SHORT", "1", "1", "1.0000", "0.002000", "inf"),
    _golden("human_focus_swing_setups", "SHORT", "3", "1", "0.0000", "-0.002000", "0.0000"),
    _golden("human_focus_m5", "ALL", "1", "3", "0.3333", "0.000000", "1.0000"),
    _golden("human_focus_m5", "ALL", "3", "1", "1.0000", "0.001000", "inf"),
    _golden("human_focus_m5", "LONG", "1", "3", "0.3333", "0.000000", "1.0000"),
    _golden("human_focus_m5", "LONG", "3", "1", "1.0000", "0.001000", "inf"),
    _golden("human_focus_m5_m5", "ALL", "1", "2", "0.5000", "0.000000", "1.0000"),
    _golden("human_focus_m5_m5", "LONG", "1", "2", "0.5000", "0.000000", "1.0000"),
    _golden("human_focus_pick", "ALL", "1", "3", "0.6667", "0.008667", "27.0000"),
    _golden("human_focus_pick", "ALL", "3", "1", "1.0000", "0.010000", "inf"),
    _golden("human_focus_pick", "ALL", "5", "1", "1.0000", "0.005000", "inf"),
    _golden("human_focus_pick", "ALL", "10", "1", "0.0000", "-0.030000", "0.0000"),
    _golden("human_focus_pick", "LONG", "1", "3", "0.6667", "0.008667", "27.0000"),
    _golden("human_focus_pick", "LONG", "3", "1", "1.0000", "0.010000", "inf"),
    _golden("human_focus_pick", "LONG", "5", "1", "1.0000", "0.005000", "inf"),
    _golden("human_focus_pick", "LONG", "10", "1", "0.0000", "-0.030000", "0.0000"),
]


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

    def test_focus_only_outcomes_aggregate_exactly_as_before(self) -> None:
        """Every row, every ORIGINAL field, in order, against the pre-change golden.

        An earlier version of this fence compared only the distinct cohort
        names, so any change to counts, returns, win rates, profit factors,
        side filters, horizons, or ordering that preserved the names would
        have passed. This is the real fence for the
        COHORT_BASE_BY_SOURCE_PREFIX rewrite, and all of that is still fenced.

        **Narrowed to the original column set on 2026-08-24 (R10.C.)** The
        rollup now also carries ground rule 10's robust half - median, trimmed
        mean, p10/p90, symbol and session counts, concentration, a
        session-block interval and the evidence label - APPENDED so every
        existing reader keeps working. Those columns are new information about
        the same rows; comparing whole dicts would make the fence fail on any
        addition, which would teach the next person to delete the fence rather
        than think about it. The additions are checked by
        `test_the_robust_columns_are_additive_and_never_replace_the_originals`
        below, so nothing is unguarded.
        """
        rows = build_human_focus_performance_rows(
            CHARACTERIZATION_OUTCOMES, updated_at=_TS
        )
        original_fields = list(CHARACTERIZATION_GOLDEN[0])
        projected = [
            {field: row[field] for field in original_fields} for row in rows
        ]
        self.assertEqual(projected, CHARACTERIZATION_GOLDEN)

    def test_the_robust_columns_are_additive_and_never_replace_the_originals(self) -> None:
        """R10.C. A bare mean is what produced `regime_pause_rw`'s -1.82R, so
        it is no longer published alone - but it IS still published, because
        hiding it would be its own dishonesty."""
        rows = build_human_focus_performance_rows(
            CHARACTERIZATION_OUTCOMES, updated_at=_TS
        )
        row = rows[0]

        # The original seven survive untouched.
        for field in ("cohort", "side", "horizon_sessions", "sample_count",
                      "win_rate", "avg_side_return", "profit_factor"):
            self.assertIn(field, row)
        # And the robust half rides beside them.
        for field in ("median_return", "trimmed_mean_return", "p10_return",
                      "p90_return", "symbols", "sessions", "top_symbol_share",
                      "ci_low", "ci_high", "ci_basis", "evidence_label",
                      "meets_n_floor"):
            self.assertIn(field, row)
        # Post-hoc rollups are discovery. Nothing here can make one a
        # confirmation, whatever its n.
        self.assertEqual(row["evidence_label"], "discovery")

    def test_a_cell_that_cannot_carry_an_interval_says_why_rather_than_printing_zero(self) -> None:
        """A blank interval with no explanation reads as an oversight. These
        characterization rows span too few sessions to bootstrap, and the row
        has to say that rather than imply a precision it does not have."""
        rows = build_human_focus_performance_rows(
            CHARACTERIZATION_OUTCOMES, updated_at=_TS
        )
        unmeasured = [row for row in rows if not row["ci_low"]]
        self.assertTrue(unmeasured, "these fixtures span too few sessions to bootstrap")
        for row in unmeasured:
            self.assertEqual(row["ci_high"], "")
            self.assertTrue(row["ci_basis"].startswith("unmeasured: "))

    def test_the_characterization_can_tell_cohorts_apart(self) -> None:
        """Sensitivity control: a new sub-cohort is visible to the fence."""
        mutated = CHARACTERIZATION_OUTCOMES + [_outcome("focus_swing_extra", "ZZZ")]
        rows = build_human_focus_performance_rows(mutated, updated_at=_TS)
        self.assertIn("human_focus_swing_extra", {row["cohort"] for row in rows})
        self.assertNotEqual(rows, CHARACTERIZATION_GOLDEN)

    def test_the_characterization_can_tell_numbers_apart(self) -> None:
        """Sensitivity control: a numeric drift is visible, not just names."""
        mutated = [dict(row) for row in CHARACTERIZATION_OUTCOMES]
        mutated[0]["h1_return"] = "0.011"
        rows = build_human_focus_performance_rows(mutated, updated_at=_TS)
        self.assertEqual(
            {row["cohort"] for row in rows},
            {row["cohort"] for row in CHARACTERIZATION_GOLDEN},
            "the name set is unchanged...",
        )
        self.assertNotEqual(rows, CHARACTERIZATION_GOLDEN, "...but the fence still trips")

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


class CanonicalCohortTests(unittest.TestCase):
    """An additive vocabulary bump must not restart a reason's record.

    v3 (2026-08-21) added "SMA incoming" and changed nothing else. The nine
    reasons that carried over are byte-identical, so grading them as fresh
    cohorts would throw away the sample for no gain. Identity is the reason's
    DEFINITION; the canonical cohort is the earliest version carrying it.

    This is a READING of the record, not a rewriting of it: pick and outcome
    rows keep the version they were captured under.
    """

    def test_an_unchanged_reason_pools_back_to_its_earliest_version(self) -> None:
        from ui.annotations.veto_cohort import canonical_veto_cohort

        self.assertEqual(canonical_veto_cohort("veto_v3_volume_dry"), "veto_v1_volume_dry")
        self.assertEqual(canonical_veto_cohort("veto_v2_volume_dry"), "veto_v1_volume_dry")
        self.assertEqual(canonical_veto_cohort("veto_v1_volume_dry"), "veto_v1_volume_dry")

    def test_rows_written_before_versioning_pool_with_v1(self) -> None:
        from ui.annotations.veto_cohort import canonical_veto_cohort

        self.assertEqual(canonical_veto_cohort("veto_volume_dry"), "veto_v1_volume_dry")

    def test_a_reason_introduced_later_keeps_its_own_cohort(self) -> None:
        """'compressed' is new in v2 and 'sma_incoming' new in v3; neither has
        an earlier twin, so neither may inherit one."""
        from ui.annotations.veto_cohort import canonical_veto_cohort

        self.assertEqual(canonical_veto_cohort("veto_v2_compressed"), "veto_v2_compressed")
        self.assertEqual(canonical_veto_cohort("veto_v3_compressed"), "veto_v2_compressed")
        self.assertEqual(
            canonical_veto_cohort("veto_v3_sma_incoming"), "veto_v3_sma_incoming"
        )

    def test_a_pre_versioning_row_pools_with_the_version_that_INTRODUCED_its_code(
        self,
    ) -> None:
        """Unversioned rows were mapped only while walking the LOWEST version,
        so a code introduced later never got an unversioned mapping at all and
        its pre-versioning picks graded as a cohort of their own forever.

        Live proof: `veto_cohort_performance.csv` carried
        `human_focus_veto_compressed` (n=3, PF 165) beside
        `human_focus_veto_v2_compressed` (n=18, PF 0.39) - one judgement read
        as two opposite ones, with the three-sample half looking spectacular.

        Version-literal-free by house rule: the vocabulary is LOADED and the
        late-introduced code is DISCOVERED, so a future bump cannot make this
        test wrong without making the behaviour wrong too.
        """
        from ui.annotations.veto_cohort import canonical_veto_cohort, veto_cohort_source
        from ui.annotations.vocabulary import available_veto_versions, load_veto_vocabulary

        versions = sorted(available_veto_versions())
        self.assertGreater(len(versions), 1, "needs at least two vocabularies")

        earliest_version_of: dict[str, int] = {}
        for version in versions:
            for reason in load_veto_vocabulary(version=version).reasons:
                earliest_version_of.setdefault(reason.code, version)

        late = {
            code: version
            for code, version in earliest_version_of.items()
            if version != versions[0]
        }
        self.assertTrue(late, "no reason is introduced after the first vocabulary")

        for code, introduced_in in late.items():
            unversioned = canonical_veto_cohort(veto_cohort_source(code))
            versioned = canonical_veto_cohort(veto_cohort_source(code, introduced_in))
            self.assertEqual(
                unversioned,
                versioned,
                f"pre-versioning {code!r} must grade with the version that introduced it",
            )
            # And it must not have been dumped on the first vocabulary, which
            # never defined the code at all.
            self.assertNotEqual(unversioned, veto_cohort_source(code, versions[0]))

    def test_every_known_code_has_an_unversioned_mapping(self) -> None:
        """No code may be left without one: a row written before the key
        carried a version is exactly the row that cannot say which vocabulary
        it meant, so leaving it unmapped is what stranded it."""
        from ui.annotations.veto_cohort import canonical_veto_cohort, veto_cohort_source
        from ui.annotations.vocabulary import available_veto_versions, load_veto_vocabulary

        for version in sorted(available_veto_versions()):
            for reason in load_veto_vocabulary(version=version).reasons:
                bare = veto_cohort_source(reason.code)
                self.assertNotEqual(
                    canonical_veto_cohort(bare),
                    bare,
                    f"{reason.code!r} has no pooled home for its pre-versioning rows",
                )

    def test_a_retired_reason_is_never_pooled_into_a_survivor(self) -> None:
        """v1's 'support_resistance_cluttered' was replaced, not renamed."""
        from ui.annotations.veto_cohort import canonical_veto_cohort

        self.assertEqual(
            canonical_veto_cohort("veto_v1_support_resistance_cluttered"),
            "veto_v1_support_resistance_cluttered",
        )

    def test_a_source_this_map_does_not_know_is_returned_unchanged(self) -> None:
        from ui.annotations.veto_cohort import canonical_veto_cohort

        self.assertEqual(canonical_veto_cohort("human_focus_auto"), "human_focus_auto")
        self.assertEqual(canonical_veto_cohort("veto_v9_invented"), "veto_v9_invented")
        self.assertEqual(canonical_veto_cohort(""), "")


class PooledPerformanceTests(unittest.TestCase):
    """The rollup groups by the canonical cohort; the outcomes keep theirs."""

    def _outcome_row(self, source: str, symbol: str, ret: str) -> dict:
        """Named ``_outcome_row``, not ``_outcome``: unittest.TestCase keeps
        its own ``_outcome`` attribute and shadowing it breaks the runner."""
        return {
            "trade_date": "2026-08-18",
            "symbol": symbol,
            "side": "LONG",
            "source": source,
            "h1_return": ret,
            "fully_matured": "",
        }

    def _write_outcomes(self, path: Path, rows: list[dict]) -> None:
        columns = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_two_versions_of_one_reason_grade_as_one_cohort(self) -> None:
        from ui.annotations.veto_cohort import _rebuild_pooled_performance

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcomes = root / "veto_cohort_outcomes.csv"
            performance = root / "veto_cohort_performance.csv"
            self._write_outcomes(
                outcomes,
                [
                    self._outcome_row("veto_v2_volume_dry", "AAA", "0.01"),
                    self._outcome_row("veto_v3_volume_dry", "BBB", "-0.02"),
                ],
            )
            written = _rebuild_pooled_performance(
                outcomes_path=outcomes, performance_path=performance, now=NOW
            )
            self.assertIsNotNone(written)
            rows = list(csv.DictReader(performance.open(encoding="utf-8")))
            cohorts = {row["cohort"] for row in rows}
            self.assertIn("human_focus_veto_v1_volume_dry", cohorts)
            self.assertNotIn("human_focus_veto_v2_volume_dry", cohorts)
            self.assertNotIn("human_focus_veto_v3_volume_dry", cohorts)
            pooled = [
                row
                for row in rows
                if row["cohort"] == "human_focus_veto_v1_volume_dry"
                and row["side"] == "ALL"
                and row["horizon_sessions"] == "1"
            ]
            self.assertEqual(len(pooled), 1)
            self.assertEqual(pooled[0]["sample_count"], "2")

    def test_the_outcome_rows_themselves_are_never_rewritten(self) -> None:
        from ui.annotations.veto_cohort import _rebuild_pooled_performance

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcomes = root / "veto_cohort_outcomes.csv"
            performance = root / "veto_cohort_performance.csv"
            self._write_outcomes(
                outcomes, [self._outcome_row("veto_v3_volume_dry", "AAA", "0.01")]
            )
            before = outcomes.read_bytes()
            _rebuild_pooled_performance(
                outcomes_path=outcomes, performance_path=performance, now=NOW
            )
            self.assertEqual(outcomes.read_bytes(), before)

    def test_nothing_to_pool_leaves_the_delegate_s_rollup_alone(self) -> None:
        """Only-canonical sources: the function reports it did nothing rather
        than rewriting an identical file."""
        from ui.annotations.veto_cohort import _rebuild_pooled_performance

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcomes = root / "veto_cohort_outcomes.csv"
            performance = root / "veto_cohort_performance.csv"
            self._write_outcomes(
                outcomes, [self._outcome_row("veto_v1_volume_dry", "AAA", "0.01")]
            )
            self.assertIsNone(
                _rebuild_pooled_performance(
                    outcomes_path=outcomes, performance_path=performance, now=NOW
                )
            )
            self.assertFalse(performance.exists())

    def test_no_outcomes_yet_is_not_an_error(self) -> None:
        from ui.annotations.veto_cohort import _rebuild_pooled_performance

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(
                _rebuild_pooled_performance(
                    outcomes_path=root / "missing.csv",
                    performance_path=root / "perf.csv",
                    now=NOW,
                )
            )
