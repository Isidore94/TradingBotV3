"""P1-7 7b: the nightly `plan_review` slot, the challenge store and the Mentor item.

Everything points at `tmp_path`: plan, history, challenge and answer files, the
day-review root and the AI store root. The live data dir is never touched.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import project_paths  # noqa: E402

SESSION = "2026-09-24"
NIGHT = datetime(2026, 9, 25, 2, 0, tzinfo=timezone.utc)
PLAN = (
    "## Goals\n- Make 2R a week.\n\n## Rules\n- No trades in the first 15 minutes.\n\n"
    "## Setups I trade\n\n## Risk\n- Max 1% per trade.\n\n## What I am testing\n\n## Decisions\n"
)


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(project_paths, "TRADING_PLAN_FILE", tmp_path / "trading_plan.md")
    monkeypatch.setattr(project_paths, "TRADING_PLAN_HISTORY_DIR", tmp_path / "trading_plan_history")
    monkeypatch.setattr(project_paths, "PLAN_CHALLENGES_FILE", tmp_path / "plan_challenges.jsonl")
    monkeypatch.setattr(project_paths, "PLAN_CHALLENGE_ANSWERS_FILE", tmp_path / "answers.jsonl")
    monkeypatch.setattr(project_paths, "PERMUTATION_REPORT_FILE", tmp_path / "permutation_report.json")
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", tmp_path / "day_review", raising=False)
    import day_review_pack

    pack = {
        "report_card": {"lines": [
            {"key": "did_well", "text": "3 of 5 reads right", "n": 5, "measured": 5,
             "source_id": "report_card:did_well"},
        ]},
        "reads": [{"read_id": "r1", "verdict": "wrong", "horizon": "m5", "source_id": "read:r1"}],
        "congruence": [{"kind": "m5_vs_d1", "text": "2 of 4 agreed", "source_id": "congruence:m5_vs_d1"}],
    }
    path = day_review_pack.pack_path(SESSION, root=tmp_path / "day_review")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(pack), encoding="utf-8")
    ai_root = tmp_path / "ai"
    ai_root.mkdir()
    (ai_root / f"measured_report_{SESSION}.json").write_text(json.dumps({
        "report_id": "abc",
        "cells": [{"cell_id": "total_profit.all", "metric": "net R", "unit": "R", "value": -2.5,
                   "n": 12, "state": "measured"}],
    }), encoding="utf-8")
    return tmp_path


def _plan(world, text=PLAN):
    (world / "trading_plan.md").write_text(text, encoding="utf-8")


def _request(challenges, calls=None):
    def fake(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return {"model": "fake-medium", "summary": {"challenges": challenges}}

    return fake


def _run(world, request, **kwargs):
    from ai_jobs import plan_review

    return plan_review.run_plan_review(
        session_date=SESSION, now=NIGHT, root=world / "day_review", ai_root=world / "ai",
        request=request, **kwargs,
    )


GOOD = {
    "plan_line": "plan:rules:1",
    "evidence": "measured:total_profit.all",
    "text": "Net R is -2.5 over 12 trades; the rule has not paid yet.",
}


# ---------------------------------------------------------------------------
# the validator
# ---------------------------------------------------------------------------
def test_an_uncited_claim_is_rejected_and_nothing_is_stored(world):
    import plan_challenges

    _plan(world)
    uncited = [
        {"plan_line": "plan:rules:1", "evidence": "", "text": "You break this rule a lot."},
        {"plan_line": "", "evidence": "measured:total_profit.all", "text": "Something is off."},
    ]
    result = _run(world, _request(uncited))

    assert result["status"] == "ok"
    assert result["extra"]["stored"] == 0
    assert result["extra"]["drop_reasons"] == {"no_evidence": 1, "no_plan_line": 1}
    assert plan_challenges.read_challenges() == []


def test_a_citation_tonight_does_not_carry_rejects_the_whole_answer(world):
    import plan_challenges

    _plan(world)
    made_up = dict(GOOD, evidence="measured:invented.cell")
    result = _run(world, _request([GOOD, made_up]))

    assert result["status"] == "failed"
    assert "does not carry" in result["reason"]
    assert plan_challenges.read_challenges() == []

    wrong_line = dict(GOOD, plan_line="plan:rules:9")
    assert _run(world, _request([wrong_line]))["status"] == "failed"


def test_more_than_three_usable_challenges_is_rejected(world):
    _plan(world)
    four = [dict(GOOD, text=f"challenge {index}") for index in range(4)]
    result = _run(world, _request(four))

    assert result["status"] == "failed"
    assert "at most 3" in result["reason"]


# ---------------------------------------------------------------------------
# the slot
# ---------------------------------------------------------------------------
def test_a_good_night_stores_cited_challenges_once(world):
    import plan_challenges

    _plan(world)
    calls: list = []
    result = _run(world, _request([GOOD], calls))

    assert result["status"] == "ok", result
    rows = plan_challenges.read_challenges()
    assert len(rows) == 1
    row = rows[0]
    assert row["plan_line"] == "plan:rules:1"
    assert row["plan_line_text"] == "No trades in the first 15 minutes."
    assert row["evidence"] == "measured:total_profit.all"
    evidence = calls[0]["evidence"]
    assert set(evidence["allowed_evidence_ids"]) == {
        f"{SESSION}/report_card:did_well", f"{SESSION}/read:r1",
        f"{SESSION}/congruence:m5_vs_d1", "measured:total_profit.all",
    }
    assert "permutation:report" not in evidence["allowed_evidence_ids"]

    again = _run(world, _request([GOOD], calls))
    assert again["status"] == "ok" and len(calls) == 1, "an unchanged night asks no model"


def test_the_permutation_report_is_read_when_it_exists(world):
    _plan(world)
    (world / "permutation_report.json").write_text(json.dumps({"families": []}), encoding="utf-8")
    calls: list = []
    _run(world, _request([], calls))

    assert "permutation:report" in calls[0]["evidence"]["allowed_evidence_ids"]


def test_no_plan_means_no_model_and_no_plan_file(world):
    calls: list = []
    result = _run(world, _request([GOOD], calls))

    assert result["status"] == "skipped"
    assert calls == []
    assert not (world / "trading_plan.md").exists()


def test_the_deterministic_half_snapshots_a_changed_plan_and_asks_nothing(world):
    import trading_plan

    _plan(world)
    calls: list = []
    result = _run(world, _request([GOOD], calls), ask=False)

    assert result["status"] == "ok"
    assert calls == []
    assert len(trading_plan.snapshots()) == 1


def test_the_slot_is_registered_as_a_stage_three_model_slot():
    from ai_jobs import runner

    slots = {slot.name: slot for slot in runner.default_slots()}
    slot = slots["plan_review"]
    assert slot.uses_model is True
    assert slot.model_free_kwargs == {"ask": False}
    assert slot.max_attempts == 2
    names = list(slots)
    assert names.index("day_review_facts") < names.index("plan_review") < names.index("improvement_ideas")


def test_no_nightly_job_answers_a_challenge():
    """Accept and reject are the trader's clicks; no ai_jobs module may call the writer."""
    for path in (SCRIPTS / "ai_jobs").glob("*.py"):
        assert "answer_challenge" not in path.read_text(encoding="utf-8"), path.name


# ---------------------------------------------------------------------------
# accept / reject / expire
# ---------------------------------------------------------------------------
def _one_challenge(world):
    import plan_challenges

    _plan(world)
    _run(world, _request([GOOD]))
    return plan_challenges.read_challenges()[0]["challenge_id"]


def test_accept_appends_a_dated_decision_and_snapshots_the_plan(world):
    import plan_challenges
    import trading_plan

    challenge = _one_challenge(world)
    before = len(trading_plan.snapshots())
    later = NIGHT + timedelta(hours=12)
    row = plan_challenges.answer_challenge(challenge, "accept", now=later)

    assert row["decision"] == "accepted"
    parsed = trading_plan.parse_plan((world / "trading_plan.md").read_text(encoding="utf-8"))
    assert len(parsed["decisions"]) == 1
    decision = parsed["decisions"][0]
    assert decision["dated"] and decision["day"] == later.date().isoformat()
    assert "No trades in the first 15 minutes." in decision["text"]
    assert len(trading_plan.snapshots()) == before + 1
    assert plan_challenges.challenge_status(challenge, now=later) == "accepted"
    with pytest.raises(plan_challenges.PlanChallengeError):
        plan_challenges.answer_challenge(challenge, "reject_disagree", now=later)


def test_reject_logs_the_reason_and_leaves_the_plan_alone(world):
    import plan_challenges

    challenge = _one_challenge(world)
    plan_before = (world / "trading_plan.md").read_text(encoding="utf-8")
    row = plan_challenges.answer_challenge(challenge, "reject_too_few", now=NIGHT + timedelta(hours=1))

    assert row["decision"] == "rejected" and row["reason"] == "reject_too_few"
    assert (world / "trading_plan.md").read_text(encoding="utf-8") == plan_before
    stored = plan_challenges.read_answers()[challenge]
    assert stored["reason"] == "reject_too_few"
    assert plan_challenges.challenge_status(challenge, now=NIGHT + timedelta(hours=2)) == "rejected"


def test_an_untouched_challenge_expires_after_seven_days(world):
    import plan_challenges

    challenge = _one_challenge(world)

    assert plan_challenges.challenge_status(challenge, now=NIGHT + timedelta(days=6, hours=23)) == "open"
    assert plan_challenges.challenge_status(challenge, now=NIGHT + timedelta(days=7)) == "expired"
    assert plan_challenges.open_challenges(NIGHT + timedelta(days=7)) == []
    assert f"plan_challenge:{challenge}" in plan_challenges.closed_keys(NIGHT + timedelta(days=7))
    with pytest.raises(plan_challenges.PlanChallengeError):
        plan_challenges.answer_challenge(challenge, "accept", now=NIGHT + timedelta(days=8))


# ---------------------------------------------------------------------------
# the Mentor item
# ---------------------------------------------------------------------------
def test_an_open_challenge_is_a_mentor_question_and_accept_writes_the_plan(world):
    import mentor_questions
    import plan_challenges
    import trading_plan

    challenge = _one_challenge(world)
    moment = NIGHT + timedelta(hours=14)
    state = {"session": "2026-09-25", "plan_challenges": plan_challenges.open_challenges(moment)}
    result = mentor_questions.pending(state, slot=None)
    asked = [subject for subject in result.asked if subject.kind == "plan_challenge"]

    assert [subject.subject_id for subject in asked] == [challenge]
    assert "accept" in asked[0].options and "reject_too_few" in asked[0].options
    assert "No trades in the first 15 minutes." in asked[0].prompt

    outcome = mentor_questions.record_answer(asked[0], {"state": "accept"}, now=moment)
    assert outcome["ok"] is True
    assert trading_plan.parse_plan((world / "trading_plan.md").read_text(encoding="utf-8"))["decisions"]

    closed = {"session": "2026-09-25", "plan_challenges": [],
              "answered": plan_challenges.closed_keys(moment), "carried": tuple(asked)}
    assert not [s for s in mentor_questions.pending(closed, slot=None).asked if s.kind == "plan_challenge"]


def test_the_plan_challenge_consumer_reads_the_decision():
    import mentor_questions

    row = {item["kind"]: item for item in mentor_questions.consumer_report()}["plan_challenge"]
    assert row["imports"] and row["reads"], row
