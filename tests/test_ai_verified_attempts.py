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


# ---------------------------------------------------------------------------
# econ brief wiring
# ---------------------------------------------------------------------------
def _econ():
    import test_econ_brief_slot as econ_tests

    return econ_tests


def _sequence(*replies):
    """A slot-style request (keyword call) answering each call with the next reply."""
    calls = []

    def _request(**kwargs):
        calls.append(kwargs)
        return {"summary": replies[min(len(calls), len(replies)) - 1], "model": "local-test"}

    _request.calls = calls
    return _request


def _econ_bad_time(pack):
    reply = _econ()._good_reply(pack)
    reply["lines"][0]["text"] = "9:15 a.m.: durable goods orders."
    return reply


def _run_econ(tmp_path, request, brief_extra=""):
    from ai_jobs import econ_brief_narration as job

    econ = _econ()
    return job.run_econ_brief(
        session_date="2026-09-24",
        out_dir=tmp_path,
        forecasts=econ._forecasts(("2026-09-24", econ.BRIEF_0924 + brief_extra)),
        request=request,
        ledger_path=tmp_path / "ledger.jsonl",
    )


def test_econ_a_rejection_then_a_pass_is_ok_on_attempt_two(tmp_path):
    pack = _econ()._pack()
    request = _sequence(_econ_bad_time(pack), _econ()._good_reply(pack))
    outcome = _run_econ(tmp_path, request)
    assert outcome["status"] == "ok", outcome
    assert "attempt 2/3" in outcome["reason"]
    assert len(request.calls) == 2
    assert (tmp_path / "2026-09-25.json").exists()


def test_econ_the_second_attempt_quotes_the_first_rejected_sentence(tmp_path):
    pack = _econ()._pack()
    request = _sequence(_econ_bad_time(pack), _econ()._good_reply(pack))
    _run_econ(tmp_path, request)
    assert len(request.calls) == 2
    first, second = (call["evidence"] for call in request.calls)
    quoted = 'Do not write: "9:15 a.m.: durable goods orders."'
    assert quoted not in first["instructions"]
    assert quoted in second["instructions"]
    # The evidence hash is the pack's: a retry never changes what the file records.
    assert first["evidence_hash"] == second["evidence_hash"]


def test_econ_three_rejections_keep_the_last_good_file_and_name_every_reason(tmp_path):
    pack = _econ()._pack()
    _run_econ(tmp_path, _sequence(_econ()._good_reply(pack)))
    before = (tmp_path / "2026-09-25.json").read_text(encoding="utf-8")
    bad = [_econ_bad_time(pack) for _ in range(3)]
    bad[1]["lines"][1]["event_ids"] = ["x9"]
    bad[2]["lines"] = bad[2]["lines"][:1]
    request = _sequence(*bad)
    outcome = _run_econ(tmp_path, request, brief_extra="\n\nExtra line.")
    assert outcome["status"] == "degraded_no_narrative"
    assert len(request.calls) == 3
    assert "attempt 1/3: a time in" in outcome["reason"]
    assert "attempt 2/3: " in outcome["reason"]
    assert "attempt 3/3: expected 3-6 lines" in outcome["reason"]
    assert (tmp_path / "2026-09-25.json").read_text(encoding="utf-8") == before


def test_econ_a_first_time_pass_makes_exactly_one_call(tmp_path):
    request = _sequence(_econ()._good_reply(_econ()._pack()))
    outcome = _run_econ(tmp_path, request)
    assert outcome["status"] == "ok"
    assert len(request.calls) == 1
    assert "attempt 1/3" in outcome["reason"]
