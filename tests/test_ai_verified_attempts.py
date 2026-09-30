"""Verified retries inside a night slot (trader 2026-09-30: spend 5080 time on better answers).

The helper asks, runs the deterministic verifier, and on a rejection asks again
with the reason folded into the evidence. The first passing reply wins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _scripted(*replies):
    """A request that answers each call with the next reply, recording the evidence."""
    seen = []

    def _request(evidence):
        seen.append(dict(evidence))
        return {"summary": replies[len(seen) - 1]}

    _request.seen = seen
    return _request


def _validate(result):
    if result["summary"] != "good":
        raise ValueError(f"reply {result['summary']!r} is not grounded")
    return "kept"


BASE = {"instructions": "Say it.", "pack": {"a": 1}}


def test_the_default_is_three_attempts():
    from ai_jobs import attempts

    assert attempts.VERIFIED_ATTEMPTS == 3


def test_a_first_time_pass_makes_exactly_one_call():
    from ai_jobs import attempts

    request = _scripted("good", "good", "good")
    out = attempts.verified_attempts(request, _validate, evidence=BASE)
    assert len(request.seen) == 1
    assert out.value == "kept" and out.attempt == 1 and out.label == "attempt 1/3"
    assert out.rejections == ()
    assert request.seen[0] == BASE


def test_a_rejection_then_a_pass_names_attempt_two_and_feeds_the_reason_back():
    from ai_jobs import attempts

    request = _scripted("bad one", "good")
    out = attempts.verified_attempts(request, _validate, evidence=BASE)
    assert out.label == "attempt 2/3"
    assert out.rejections == ("reply 'bad one' is not grounded",)
    assert "reply 'bad one' is not grounded" in request.seen[1]["instructions"]
    assert "An earlier reply was rejected" in request.seen[1]["instructions"]
    # The first call saw the evidence untouched, and the base is never mutated.
    assert request.seen[0] == BASE and BASE["instructions"] == "Say it."


def test_three_rejections_raise_with_every_reason():
    from ai_jobs import attempts

    request = _scripted("r1", "r2", "r3", "good")
    with pytest.raises(attempts.AttemptsRejected) as caught:
        attempts.verified_attempts(request, _validate, evidence=BASE)
    assert len(request.seen) == 3
    assert len(caught.value.reasons) == 3
    text = str(caught.value)
    for index, name in enumerate(("r1", "r2", "r3"), 1):
        assert f"attempt {index}/3: reply '{name}' is not grounded" in text
    # Each retry carries every rejection so far.
    assert "'r1'" in request.seen[2]["instructions"] and "'r2'" in request.seen[2]["instructions"]


def test_a_failed_model_call_is_not_retried():
    from ai_jobs import attempts

    calls = []

    def _boom(evidence):
        calls.append(evidence)
        raise RuntimeError("endpoint down")

    with pytest.raises(RuntimeError, match="endpoint down"):
        attempts.verified_attempts(_boom, _validate, evidence=BASE)
    assert len(calls) == 1


def test_a_closed_gate_stops_the_retries_and_says_why():
    from ai_jobs import attempts

    request = _scripted("bad", "good")
    with pytest.raises(attempts.AttemptsRejected) as caught:
        attempts.verified_attempts(
            request, _validate, evidence=BASE, may_retry=lambda: (False, "window closed")
        )
    assert len(request.seen) == 1
    assert "no attempt 2/3 (window closed)" in str(caught.value)
