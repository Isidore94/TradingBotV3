"""Greatness Monitor lifecycle tests (HIGH_CONVICTION_FEATURES_PLAN sec 18)."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from greatness_monitor import (  # noqa: E402
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
