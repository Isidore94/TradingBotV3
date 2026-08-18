"""R6(b): the replay characterization fixture for `_load_resolved_events`.

`technical_integrity_events.jsonl` is append-only, and on every boot the
monitor re-reads it and reconstructs in-memory state from it. That
reconstruction - not the scoring math - is what any future change to the file's
shape would put at risk, and nothing pinned it.

`tests/fixtures/technical_integrity_scoring_v1.json` does NOT cover this. It
feeds inline events into the aggregation and asserts numbers; it never touches a
file, never calls the replay, and would stay green through a change that
corrupted every field below.

**This is a characterization fixture** (decision 0009): it records what the code
does today, not what it ought to do. Its whole job is to let a later change -
the per-session segmentation the R6(b) decision commits to, when warehouse
Phase-3 retention unlocks - prove it changed nothing. A deliberate change here
must be declared in the fixture's `intentional_difference` field, never quietly
regenerated.

The decision itself (plan.md item 6(b), 2026-08-17) is that the live file is NOT
rotated now: replay is session-filtered, so closed sessions are inert, and the
2.2 s boot re-parse does not justify breaking the warehouse ingest watermark.
This fixture is what makes the eventual segmentation checkable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from conftest import load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import technical_integrity as ti  # noqa: E402

FIXTURE_NAME = "technical_integrity_replay_v1"


@pytest.fixture(scope="module")
def contract():
    return load_fixture_contract(FIXTURE_NAME)


def replay(tmp_path, lines, state_seed):
    """Reconstruct state from ledger bytes, exactly as a boot does."""
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text(json.dumps(state_seed), encoding="utf-8")

    monitor = ti.TechnicalIntegrityMonitor(
        events_path=events,
        state_path=state,
        snapshot_path=tmp_path / "snapshot.json",
    )
    return {
        "session_date": monitor.session_date,
        "seen_test_ids": sorted(monitor.seen_test_ids),
        "resolved_event_ids": [str(row.get("event_id")) for row in monitor.resolved_events],
        "pending_ids": sorted(monitor.pending),
        "pending_t_B_keys": sorted(monitor.pending.get("t-B", {})),
        "pending_followup_ids": sorted(monitor.pending_followups),
        "pending_followup_horizons": {
            key: list(value.get("completed_horizons") or [])
            for key, value in sorted(monitor.pending_followups.items())
        },
        "followup_event_ids": sorted(monitor.followup_event_ids),
        "frozen_snapshot_markers": sorted(monitor.frozen_snapshot_markers),
        "latest_completed_bar_end": monitor.latest_completed_bar_end,
    }


@pytest.fixture
def actual(tmp_path, contract):
    return replay(tmp_path, contract["ledger_lines"], contract["state_seed"])


class TestTheContractItself:
    def test_the_fixture_states_its_own_provenance(self, contract):
        assert contract.schema == "technical_integrity_replay_fixture_v1"
        assert contract.feature_version == ti.STATE_SCHEMA
        assert contract.raw_input_digest() == contract["raw_input_sha256"]

    def test_the_configuration_still_matches_the_module(self, contract):
        config = contract.configuration
        assert tuple(config["followup_horizons_minutes"]) == ti.FOLLOWUP_HORIZONS_MINUTES
        assert set(config["append_time_provenance_fields"]) == set(
            ti._APPEND_TIME_PROVENANCE_FIELDS
        )


class TestTheReplayIsUnchanged:
    def test_the_whole_reconstruction_matches(self, contract, actual):
        contract.assert_matches(actual, contract["expected"], "monolith replay")

    def test_a_started_resolved_pair_resolves_rather_than_pends(self, actual):
        assert "t-A" in actual["resolved_event_ids"]
        assert "t-A" not in actual["pending_ids"]

    def test_an_unresolved_start_recovers_into_pending(self, actual):
        assert "t-B" in actual["pending_ids"]

    def test_recovery_strips_append_time_provenance(self, actual):
        """A restart between touch and resolution must not change the row.

        `_append_event` stamps as_of/written_at onto what it writes, so a ledger
        row carries the STARTED event's provenance. Recovering it verbatim gave
        the later resolution the touch time as its as_of.
        """
        assert "as_of" not in actual["pending_t_B_keys"]
        assert "written_at" not in actual["pending_t_B_keys"]
        assert "event_id" in actual["pending_t_B_keys"]

    def test_a_resolution_suppresses_stale_seeded_pending(self, actual):
        """t-C is resolved in the ledger and pending in the state seed."""
        assert "t-C" in actual["resolved_event_ids"]
        assert "t-C" not in actual["pending_ids"]

    def test_seeded_pending_with_no_ledger_row_survives(self, actual):
        assert "t-Z" in actual["pending_ids"]

    def test_resolved_events_sort_by_time_then_event_id(self, contract, actual):
        """t-D and t-E share a resolved_at and are WRITTEN in reverse order."""
        order = actual["resolved_event_ids"]
        assert order.index("t-D") < order.index("t-E")
        assert order == contract["expected"]["resolved_event_ids"]

    def test_a_partial_followup_chain_stays_pending(self, actual):
        assert actual["pending_followup_horizons"]["f-1"] == [30, 60]

    def test_a_complete_followup_chain_drops(self, actual):
        assert "f-2" not in actual["pending_followup_ids"]
        # ...but its rows are still counted as seen, which is deliberate:
        # dropping from pending is not the same as never having happened.
        assert {"fu-3", "fu-4", "fu-5"} <= set(actual["followup_event_ids"])

    def test_all_four_snapshot_marker_types_are_collected(self, actual):
        assert actual["frozen_snapshot_markers"] == [
            "sk-frozen", "sk-missed", "sk-missed-orb", "sk-orb",
        ]

    def test_a_truncated_final_line_costs_that_line_only(self, contract, actual):
        """A crash mid-flush must not make a day of evidence unreadable."""
        assert contract["ledger_lines"][-1].startswith("{")
        with pytest.raises(json.JSONDecodeError):
            json.loads(contract["ledger_lines"][-1])
        assert actual["resolved_event_ids"]


class TestCrossSessionRowsAreInert:
    """The property the whole R6(b) decision rests on."""

    def test_no_other_session_leaks_into_the_reconstruction(self, actual):
        everything = (
            set(actual["seen_test_ids"])
            | set(actual["resolved_event_ids"])
            | set(actual["pending_ids"])
            | set(actual["pending_followup_ids"])
            | set(actual["followup_event_ids"])
            | set(actual["frozen_snapshot_markers"])
        )
        assert not {name for name in everything if name.startswith(("x-", "n-"))}

    def test_the_watermark_ignores_a_later_session(self, contract, actual):
        """Non-vacuous: the later rows sort ABOVE every current-session row.

        A prior session alone could not test this - its as_of sorts below, so
        deleting the session filter would leave the watermark unchanged and the
        assertion would pass while measuring nothing.
        """
        later = contract.configuration["next_session_date"]
        assert actual["latest_completed_bar_end"].startswith(
            contract.configuration["session_date"]
        )
        assert not actual["latest_completed_bar_end"].startswith(later)

    def test_the_same_bytes_replay_differently_for_a_later_session(
        self, tmp_path, contract
    ):
        """The positive control.

        Without it, "the filter excludes those rows" and "that field is never
        reachable at all" look identical from the outside.
        """
        seed = dict(contract["state_seed"])
        seed["session_date"] = contract.configuration["next_session_date"]
        seed["pending"] = {}
        seed["pending_followups"] = {}
        seed["seen_test_ids"] = []

        control = replay(tmp_path, contract["ledger_lines"], seed)

        contract.assert_matches(
            control, contract["expected_next_session"], "next-session replay"
        )
        assert control["latest_completed_bar_end"].startswith(
            contract.configuration["next_session_date"]
        )


class TestSegmentationWouldChangeNothing:
    """What makes the eventual per-session segmentation checkable.

    The R6(b) decision commits the shape of the future retention work: closed
    sessions become immutable per-session files and the monolith freezes in
    place. This test is the claim that shape rests on - replaying only the
    current session's segment must reconstruct byte-identical state.
    """

    def test_the_segments_partition_the_ledger(self, contract):
        segments = contract["segment_lines"]
        total = sum(len(lines) for lines in segments.values())
        assert total == len(contract["ledger_lines"])

    def test_replaying_one_segment_equals_replaying_the_monolith(
        self, tmp_path, contract, actual
    ):
        session = contract.configuration["session_date"]
        segmented = replay(
            tmp_path, contract["segment_lines"][session], contract["state_seed"]
        )

        contract.assert_matches(segmented, contract["expected_segmented"], "segmented")
        assert segmented == actual
