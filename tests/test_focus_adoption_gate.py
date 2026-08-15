"""The combined M5 Focus adoption gate (plan.md Phase 0.5, packet R2 Part A).

Trader rule 2026-08-14: an auto M5 Focus pick must be **above the previous
day's high AND above session VWAP** on the M5 for longs, and below both for
shorts. The same test runs at candidate build, on every staging refresh (a
pick that falls back through either level is evicted), and again at adoption.

This file opens with the golden characterization fixture required by
plan.md sec 5 before any detector/routing change: it freezes what the
candidate filter selected BEFORE the VWAP half existed, so the gate's effect
on selection is a reviewable diff rather than an assertion nobody can check.
"""

import sys
from pathlib import Path

from conftest import load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURE_NAME = "auto_pick_focus_gate_v1"


def _filtered(fixture):
    import autopilot_core as core

    result = core.filter_candidates_by_prev_day_extremes(
        fixture["candidates"], fixture["profiles"], fixture["daily_context"]
    )
    return {
        side: sorted(row["symbol"] for row in result[side])
        for side in ("longs", "shorts")
    }


def test_candidate_filter_golden_fixture():
    """Loading re-verifies raw_input_sha256 over the fixture's own inputs, so
    editing a profile without re-freezing the expectations fails here."""
    fixture = load_fixture_contract(FIXTURE_NAME)
    assert fixture.schema == "auto_pick_focus_gate_v1"
    fixture.assert_matches(_filtered(fixture), fixture["expected"], "candidate filter")
