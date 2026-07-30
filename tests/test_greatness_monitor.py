"""Greatness Monitor lifecycle tests (plan.md sections 7.3, 9, and 10)."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from conftest import load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from greatness_monitor import (  # noqa: E402
    PROVENANCE_SCHEMA,
    Condition,
    ConfirmationPlan,
    ConfirmationStep,
    DevelopmentCandidate,
    EventType,
    GreatnessEngine,
    Stage,
    candidate_from_d1_trigger_levels,
)
from market_state import M5Bar, mirror_bar  # noqa: E402

START = datetime(2026, 7, 13, 9, 35)


def bar(i, open_, high, low, close, complete=True):
    return M5Bar(
        ts=START + timedelta(minutes=5 * i),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1_000_000,
        complete=complete,
    )


def make_candidate(levels=(101.0,), condition=Condition.CLOSE, invalidation=98.0, **rearm):
    steps = [ConfirmationStep(label=f"L{i}", level=lv, condition=condition) for i, lv in enumerate(levels, 1)]
    plan = ConfirmationPlan(side_sign=1, steps=steps, invalidation=invalidation)
    if rearm:
        plan.rearm.max_attempts = rearm.get("max_attempts", plan.rearm.max_attempts)
        plan.rearm.min_reset_bars = rearm.get("min_reset_bars", plan.rearm.min_reset_bars)
    return DevelopmentCandidate(
        symbol="NVDA", side="LONG", setup_family="test", session_date="2026-07-13", plan=plan
    )


def run(candidate, bars, engine=None):
    engine = engine or GreatnessEngine()
    events = []
    for b in bars:
        events.extend(engine.on_bar(candidate, b))
    return events


def kinds(events):
    return [e.event for e in events]


def test_wick_through_is_evidence_not_confirmation():
    cand = make_candidate()
    events = run(cand, [bar(0, 100.0, 101.4, 99.8, 100.4)])  # wick over 101, close below
    assert EventType.LEVEL_TOUCHED in kinds(events)
    assert EventType.WICK_THROUGH in kinds(events)
    assert EventType.FAILED_ATTEMPT in kinds(events)
    assert EventType.READY not in kinds(events)
    assert cand.stage == Stage.FAILED and cand.attempts == 1


def test_close_through_makes_ready_exactly_once():
    cand = make_candidate()
    events = run(cand, [bar(0, 100.0, 101.6, 99.9, 101.3), bar(1, 101.3, 101.9, 101.0, 101.7)])
    assert kinds(events).count(EventType.READY) == 1
    assert EventType.CLOSED_THROUGH in kinds(events)
    assert cand.stage == Stage.READY
    assert cand.readiness == 1.0


def test_failed_attempt_rearms_and_later_confirmation_still_fires():
    cand = make_candidate(min_reset_bars=2, max_attempts=2)
    bars = [
        bar(0, 100.0, 101.4, 99.8, 100.3),   # wick fail -> FAILED
        bar(1, 100.3, 100.6, 99.9, 100.2),   # reset bar 1
        bar(2, 100.2, 100.5, 99.9, 100.4),   # reset bar 2 -> REARMED
        bar(3, 100.4, 101.8, 100.3, 101.5),  # genuine close through -> READY
    ]
    events = run(cand, bars)
    assert EventType.REARMED in kinds(events)
    assert EventType.READY in kinds(events)
    assert cand.stage == Stage.READY


def test_max_attempts_exhausts_rearming():
    cand = make_candidate(min_reset_bars=1, max_attempts=1)
    bars = [
        bar(0, 100.0, 101.4, 99.8, 100.3),  # attempt 1 fails -> attempts == max
        bar(1, 100.3, 100.6, 99.9, 100.2),
        bar(2, 100.2, 101.9, 100.1, 101.6),  # would confirm, but no attempts left
    ]
    events = run(cand, bars)
    assert EventType.REARMED not in kinds(events)
    assert EventType.READY not in kinds(events)
    assert cand.stage == Stage.FAILED


def test_multi_step_ladder_tracks_readiness():
    cand = make_candidate(levels=(101.0, 103.0))
    events = run(cand, [bar(0, 100.0, 101.7, 99.9, 101.4)])
    assert cand.readiness == 0.5
    assert cand.stage == Stage.CONFIRMING
    assert EventType.READY not in kinds(events)
    events = run(cand, [bar(1, 101.4, 103.6, 101.2, 103.3)])
    assert cand.readiness == 1.0
    assert EventType.READY in kinds(events)


def test_acceptance_requires_consecutive_closes():
    cand = make_candidate(condition=Condition.ACCEPT)
    cand.plan.steps[0].required_bars = 2
    events = run(cand, [bar(0, 100.0, 101.5, 99.9, 101.2)])
    assert EventType.ACCEPTED not in kinds(events)
    assert cand.stage == Stage.CONFIRMING
    events = run(cand, [bar(1, 101.2, 101.8, 101.0, 101.5)])
    assert EventType.ACCEPTED in kinds(events)
    assert cand.stage == Stage.READY


def test_invalidation_is_terminal():
    cand = make_candidate()
    events = run(cand, [bar(0, 100.0, 100.5, 97.5, 97.8)])
    assert kinds(events) == [EventType.INVALIDATED]
    assert cand.stage == Stage.INVALIDATED
    assert run(cand, [bar(1, 97.8, 102.5, 97.5, 102.0)]) == []


def test_incomplete_bar_is_a_noop():
    cand = make_candidate()
    events = run(cand, [bar(0, 100.0, 101.9, 99.9, 101.5, complete=False)])
    assert events == []
    assert cand.stage == Stage.DISCOVERED


def test_restart_round_trip_preserves_progression():
    cand = make_candidate(levels=(101.0, 103.0))
    run(cand, [bar(0, 100.0, 101.7, 99.9, 101.4)])  # clears L1
    restored = DevelopmentCandidate.from_dict(cand.to_dict())
    assert restored.stage == cand.stage
    assert restored.readiness == 0.5
    events = run(restored, [bar(1, 101.4, 103.6, 101.2, 103.3)])
    assert EventType.READY in kinds(events)


def test_short_mirror_equivalence():
    pivot = 100.0
    long_cand = make_candidate(levels=(101.0,), invalidation=98.0)
    long_bars = [bar(0, 100.0, 101.4, 99.8, 100.3), bar(1, 100.3, 101.8, 100.2, 101.5)]
    long_events = run(long_cand, long_bars)

    short_steps = [ConfirmationStep(label="L1", level=2 * pivot - 101.0, condition=Condition.CLOSE)]
    short_cand = DevelopmentCandidate(
        symbol="NVDA",
        side="SHORT",
        setup_family="test",
        session_date="2026-07-13",
        plan=ConfirmationPlan(side_sign=-1, steps=short_steps, invalidation=2 * pivot - 98.0),
    )
    short_events = run(short_cand, [mirror_bar(b, pivot) for b in long_bars])
    assert kinds(short_events) == kinds(long_events)
    assert short_cand.stage == long_cand.stage
    assert short_cand.attempts == long_cand.attempts


def test_d1_adapter_builds_ordered_plan():
    rows = [
        {"label": "UPPER_2", "level": 105.0, "setup_family": "mid_earnings"},
        {"label": "UPPER_1", "level": 101.0},
        {"label": "DUP", "level": 101.0},
    ]
    cand = candidate_from_d1_trigger_levels(
        "nvda", "LONG", rows, session_date="2026-07-13", invalidation=97.0
    )
    assert cand is not None
    assert [s.label for s in cand.plan.steps] == ["UPPER_1", "UPPER_2"]  # nearest first
    assert cand.setup_family == "mid_earnings"
    assert cand.plan.invalidation == 97.0
    assert candidate_from_d1_trigger_levels("X", "LONG", [], session_date="2026-07-13") is None


# ---------------------------------------------------------------------------
# Golden characterization of the D1 adapter (plan.md Milestone 3 / sec 5).
#
# candidate_from_d1_trigger_levels is detector-adjacent: it decides which armed
# levels become confirmation steps, in what order, under which label, and which
# ones collapse away.  This fixture freezes that decision byte-for-byte so any
# later work on the adapter (provenance plumbing included) has to prove it is
# additive.  If this test fails, the adapter's behaviour changed - fix the code,
# never the fixture.
# ---------------------------------------------------------------------------

D1_ADAPTER_FIXTURE = "greatness_candidate_from_d1_v1"

#: The exact surface the fixture pins.  Deliberately an explicit whitelist, not
#: ``DevelopmentCandidate.to_dict()``: additive metadata (provenance) must be
#: free to appear on the candidate without silently rewriting this golden.
_PINNED_STEP_KEYS = (
    "label",
    "level",
    "condition",
    "required_bars",
    "mandatory",
    "cleared",
    "cleared_at",
    "accept_progress",
    "closed_through",
)


def _project_candidate(candidate):
    """Project a candidate down to the pinned detector output."""
    if candidate is None:
        return None
    plan = candidate.plan
    return {
        "symbol": candidate.symbol,
        "side": candidate.side,
        "setup_family": candidate.setup_family,
        "session_date": candidate.session_date,
        "stage": candidate.stage.value,
        "attempts": candidate.attempts,
        "readiness": candidate.readiness,
        "plan": {
            "side_sign": plan.side_sign,
            "version": plan.version,
            "invalidation": plan.invalidation,
            "obstacle": plan.obstacle,
            "target": plan.target,
            "rearm": {
                "max_attempts": plan.rearm.max_attempts,
                "min_reset_bars": plan.rearm.min_reset_bars,
            },
            "steps": [
                {key: getattr(step, key) for key in _PINNED_STEP_KEYS}
                | {"condition": step.condition.value}
                for step in plan.steps
            ],
        },
    }


def _d1_adapter_case_names():
    contract = load_fixture_contract(D1_ADAPTER_FIXTURE)
    return sorted(contract["trigger_rows"])


@pytest.mark.parametrize("case_name", _d1_adapter_case_names())
def test_d1_adapter_matches_golden_fixture(case_name):
    contract = load_fixture_contract(D1_ADAPTER_FIXTURE)
    assert contract.schema == D1_ADAPTER_FIXTURE
    assert contract.feature_version == "greatness_v1"
    # the fixture's declared configuration is the engine's real configuration
    assert contract.configuration["engine_version"] == "greatness_v1"
    assert contract.configuration["default_confirmation"] == Condition.CLOSE.value
    assert contract.configuration["rearm_max_attempts"] == ConfirmationPlan(side_sign=1).rearm.max_attempts
    assert contract.configuration["rearm_min_reset_bars"] == ConfirmationPlan(side_sign=1).rearm.min_reset_bars

    case = contract["trigger_rows"][case_name]
    produced = candidate_from_d1_trigger_levels(
        case["symbol"],
        case["side"],
        case["rows"],
        session_date=case["session_date"],
        setup_family=case["setup_family"],
        invalidation=case["invalidation"],
        target=case["target"],
        confirmation=Condition(case["confirmation"]),
    )
    actual = _project_candidate(produced)
    expected = contract["expected"][case_name]["candidate"]

    contract.assert_matches(actual, expected, context=case_name)
    # byte-identical, not merely within tolerance
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


# ---------------------------------------------------------------------------
# W07: real D1 trigger provenance on the candidate (plan.md sec 7.3).
# ---------------------------------------------------------------------------

def test_d1_adapter_carries_real_trigger_provenance():
    contract = load_fixture_contract(D1_ADAPTER_FIXTURE)
    case = contract["trigger_rows"]["long_ordered_ladder"]
    cand = candidate_from_d1_trigger_levels(
        case["symbol"],
        case["side"],
        case["rows"],
        session_date=case["session_date"],
        invalidation=case["invalidation"],
        target=case["target"],
    )
    assert cand is not None

    provenance = cand.provenance
    assert provenance["schema"] == PROVENANCE_SCHEMA
    assert provenance["source"] == "d1_trigger_levels"
    assert provenance["primary_trigger_id"] == "avwape_reclaim:AVWAPE:101.5000"
    assert provenance["trigger_ids"] == [
        "avwape_reclaim:AVWAPE:101.5000",
        "second_dev_break:UPPER_2:105.2500",
    ]
    assert provenance["event_types"] == ["avwape_reclaim", "second_dev_break"]
    assert provenance["priority_buckets"] == ["A", "B"]
    assert provenance["sources"] == ["master_avwap_d1"]
    assert provenance["target_tiers"] == ["A/S"]
    assert provenance["source_schema_versions"] == [1]
    assert provenance["armed_at"] == "2026-07-13"
    assert provenance["upgrade_only"] is True
    assert provenance["rows_total"] == 3
    assert provenance["rows_used"] == 2
    assert provenance["rows_dropped_no_level"] == 0
    assert provenance["rows_dropped_duplicate_level"] == 1
    assert provenance["steps_missing_trigger_id"] == 0

    # every replay field the upstream row carried is on the step it produced
    step = cand.plan.steps[0]
    assert step.provenance["trigger_id"] == "avwape_reclaim:AVWAPE:101.5000"
    assert step.provenance["event_type"] == "avwape_reclaim"
    assert step.provenance["reason"] == "reclaim of the earnings anchored VWAP"
    assert step.provenance["source"] == "master_avwap_d1"
    assert step.provenance["anchor_type"] == "earnings"
    assert step.provenance["anchor_date"] == "2026-04-24"
    assert step.provenance["armed_at"] == "2026-07-13"
    assert step.provenance["armed_price"] == 99.8
    assert step.provenance["action"] == "break_above"
    assert step.provenance["alert_label"] == "AVWAP-E reclaim"
    assert step.provenance["target_tier"] == "A/S"
    assert step.provenance["upgrade_only"] is True
    # the level this step collapsed is not lost
    assert step.provenance["superseded_trigger_ids"] == ["first_dev_break:UPPER_1:101.5000"]


def test_d1_provenance_reports_dropped_rows_and_missing_ids():
    contract = load_fixture_contract(D1_ADAPTER_FIXTURE)
    case = contract["trigger_rows"]["label_fallbacks_and_dropped_rows"]
    cand = candidate_from_d1_trigger_levels(
        case["symbol"], case["side"], case["rows"], session_date=case["session_date"]
    )
    assert cand is not None
    provenance = cand.provenance
    assert provenance["rows_total"] == 4
    assert provenance["rows_used"] == 2
    assert provenance["rows_dropped_no_level"] == 2
    assert provenance["rows_dropped_duplicate_level"] == 0
    assert provenance["steps_missing_trigger_id"] == 0
    # a row that never declared upgrade_only leaves the aggregate uncertain
    assert provenance["upgrade_only"] is True


def test_rows_without_trigger_ids_are_counted_never_invented():
    cand = candidate_from_d1_trigger_levels(
        "NVDA",
        "LONG",
        [{"label": "UPPER_1", "level": 101.0, "setup_family": "test"}],
        session_date="2026-07-13",
    )
    assert cand is not None
    assert cand.plan.steps[0].provenance["trigger_id"] == ""
    assert cand.provenance["primary_trigger_id"] == ""
    assert cand.provenance["steps_missing_trigger_id"] == 1
    assert cand.provenance["upgrade_only"] is None


def test_provenance_survives_the_restart_round_trip():
    contract = load_fixture_contract(D1_ADAPTER_FIXTURE)
    case = contract["trigger_rows"]["short_ordered_ladder"]
    cand = candidate_from_d1_trigger_levels(
        case["symbol"], case["side"], case["rows"], session_date=case["session_date"]
    )
    restored = DevelopmentCandidate.from_dict(cand.to_dict())
    assert restored.provenance == cand.provenance
    assert [s.provenance for s in restored.plan.steps] == [s.provenance for s in cand.plan.steps]


def test_pre_provenance_candidate_payload_loads_with_empty_blocks():
    """A stored candidate written before W07 must still restore cleanly."""
    cand = make_candidate()
    payload = cand.to_dict()
    payload.pop("provenance")
    for step in payload["plan"]["steps"]:
        step.pop("provenance")

    restored = DevelopmentCandidate.from_dict(payload)

    assert restored.provenance == {}
    assert all(step.provenance == {} for step in restored.plan.steps)
    assert restored.stage == cand.stage
