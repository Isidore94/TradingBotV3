"""Packet N2 - the synthesis JSON stops shearing at 3,500 tokens.

Two nights of four (2026-09-02 and 2026-09-04) published UNSYNTHESIZED because
the reduce call's answer was cut mid-string at ~14,500 chars - the local
provider hitting the one hard-coded ``max_tokens`` of 3,500 that the map slices
and the synthesis share. The retry then re-sent the identical request with the
validator's rejection appended, which is more prompt and the same ceiling, so
it cut again and ~7 minutes of generation were spent proving it.

Everything here drives the REAL local path - ``run_map_reduce`` ->
``ai_summary.request_ai_summary`` -> ``_request_local_summary`` - with a fake
``post``. No model is ever called. The fixtures for the local provider live in
``tests/test_local_ai_provider.py`` rather than in ``test_ai_summary.py`` or
``test_ai_map_reduce.py`` (neither of those two plumbs a ``post`` into the
local branch at all), so the small amount that is needed is restated here
rather than imported across test modules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import map_reduce  # noqa: E402

ENDPOINT = "http://127.0.0.1:11434/v1"

#: What a length stop actually looks like on the wire: a valid JSON PREFIX that
#: simply stops. This is the 2026-09-04 failure in miniature - the parser's
#: complaint was "Unterminated string starting at: line 1 column 14709".
CUT_JSON = (
    '{"executive_summary": "The selected evidence supports one measured finding.", '
    '"what_is_working": [{"statement": "Swing rows are shown fir'
)


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


def _chat(text, *, finish_reason="stop", done_reason=None):
    """An OpenAI-compatible chat-completions body.

    ``finish_reason`` is on the choice, where llama.cpp and every
    OpenAI-compatible server put it; ``done_reason`` is Ollama's native field
    and is only present when asked for. ``usage`` is deliberately absent so the
    prompt-truncation tripwire (a different failure) stays out of these tests.
    """
    choice = {"index": 0, "message": {"role": "assistant", "content": text}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    payload = {"id": "chatcmpl-local", "object": "chat.completion", "choices": [choice]}
    if done_reason is not None:
        payload["done_reason"] = done_reason
    return _Response(payload)


def _settings(**values):
    """Patch the local-settings reader for scripts/ai_summary.py."""
    import ai_summary

    return mock.patch.object(
        ai_summary,
        "get_local_setting",
        lambda key, default=None: values.get(key, default),
    )


def _package(sources):
    return {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": "2026-09-05T02:00:00-07:00",
        "session_date": "2026-09-04",
        "sources": [
            {
                "source_id": sid,
                "label": sid,
                "status": "available",
                "content": content,
            }
            for sid, content in sources
        ],
    }


def _summary(statement, refs, section="what_is_working", confidence="medium"):
    base = {
        name: []
        for name in (
            "what_is_working",
            "what_is_not_working",
            "best_candidates",
            "lessons_for_tomorrow",
            "risk_notes",
        )
    }
    base[section] = [
        {"statement": statement, "evidence_refs": list(refs), "confidence": confidence}
    ]
    return {"executive_summary": "One measured finding.", **base}


def _user_content(kwargs) -> str:
    return str(kwargs["json"]["messages"][1]["content"])


def _is_reduce_call(kwargs) -> bool:
    """The reduce call is the one holding the findings source."""
    return map_reduce.FINDINGS_SOURCE_ID in _user_content(kwargs)


def _map_reply(kwargs):
    """A valid map answer citing the ONE source that slice was handed."""
    source_id = "b.two" if "b.two" in _user_content(kwargs) else "a.one"
    return _chat(json.dumps(_summary(f"a finding from {source_id}", [source_id])))


def _run(fake_post, *, sources=(("a.one", ["x"]), ("b.two", ["y"]))):
    """``run_map_reduce`` over the real ``request_ai_summary`` local branch."""
    import ai_summary

    def call(**kwargs):
        return ai_summary.request_ai_summary(post=fake_post, **kwargs)

    with _settings(ai_local_endpoint_url=ENDPOINT):
        return map_reduce.run_map_reduce(
            evidence=_package(list(sources)), model="gemma3:12b", request=call
        )


# ------------------------------------------------------------------ item 1


def test_synthesis_call_sends_its_own_cap():
    """The reduce call gets the SYNTHESIS cap; a map slice keeps the map cap.

    Both were 3,500 on e7b12ebe, which is what sheared the document.
    """
    import ai_summary

    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs)
        if _is_reduce_call(kwargs):
            return _chat(json.dumps(_summary("synthesized", [map_reduce.FINDINGS_SOURCE_ID])))
        # A map slice may cite only the one source it was handed.
        source_id = "b.two" if "b.two" in _user_content(kwargs) else "a.one"
        return _chat(json.dumps(_summary(f"a finding from {source_id}", [source_id])))

    result = _run(fake_post)
    assert result["map_reduce"]["synthesized"] is True

    map_cap = getattr(ai_summary, "LOCAL_MAP_GENERATION_TOKENS", None)
    synthesis_cap = getattr(ai_summary, "LOCAL_SYNTHESIS_GENERATION_TOKENS", None)
    assert isinstance(map_cap, int) and isinstance(synthesis_cap, int), (
        "the map and synthesis caps must be two separate constants; got "
        f"LOCAL_MAP_GENERATION_TOKENS={map_cap!r} "
        f"LOCAL_SYNTHESIS_GENERATION_TOKENS={synthesis_cap!r}"
    )
    assert synthesis_cap > map_cap, (
        f"the synthesis cap must be larger than the map cap; got {synthesis_cap} vs {map_cap}"
    )

    reduce_calls = [c for c in calls if _is_reduce_call(c)]
    map_calls = [c for c in calls if not _is_reduce_call(c)]
    assert len(reduce_calls) == 1 and len(map_calls) == 2

    assert reduce_calls[0]["json"]["max_tokens"] == synthesis_cap, (
        "the synthesis request still sends the map cap: "
        f"{reduce_calls[0]['json']['max_tokens']}"
    )
    for call in map_calls:
        assert call["json"]["max_tokens"] == map_cap, (
            f"a map slice sent {call['json']['max_tokens']}, not the map cap {map_cap}"
        )


def test_budget_unchanged_by_the_synthesis_cap():
    """The evidence budget keeps subtracting the MAP cap.

    The reduce prompt is the model's own findings, not evidence, and the 64k
    window holds 8k of output beside them - so widening the OUTPUT cap must not
    narrow the evidence every map slice is allowed to carry. Computed from the
    module's own constants, never a remembered number (the 2026-08-27 shear was
    a hand-carried number in a comment).
    """
    import ai_summary

    map_cap = getattr(ai_summary, "LOCAL_MAP_GENERATION_TOKENS", None)
    assert map_cap == 3_500, f"the map cap stays 3,500; got {map_cap!r}"
    # The old name must survive: tests/test_local_ai_provider.py imports it.
    assert ai_summary.LOCAL_GENERATION_TOKENS == map_cap

    for context in (ai_summary.DEFAULT_LOCAL_CONTEXT_TOKENS, 65_536):
        expected = max(
            1_000,
            int(
                (context - map_cap)
                * ai_summary._BUDGET_CHARS_PER_TOKEN
                * ai_summary._BUDGET_RETRY_HEADROOM
                / ai_summary._BUDGET_PROMPT_OVERHEAD
            ),
        )
        with _settings(ai_local_context_tokens=context, ai_local_evidence_budget_chars=0):
            assert ai_summary.local_evidence_budget_ceiling_chars() == expected
            assert ai_summary.local_evidence_budget_chars() == expected


# ------------------------------------------------------------------ item 2


def test_length_stop_retries_shorter_not_identical():
    """A length cut is not malformed output a model can repair.

    Re-sending the identical request plus the rejection is MORE prompt against
    the same ceiling, so it cuts again. The one retry must ask for a shorter
    answer instead, and the manifest must say it did.
    """
    reduce_calls = []

    def fake_post(url, **kwargs):
        if not _is_reduce_call(kwargs):
            return _map_reply(kwargs)
        reduce_calls.append(kwargs)
        if len(reduce_calls) == 1:
            return _chat(CUT_JSON, finish_reason="length")
        return _chat(json.dumps(_summary("synthesized", [map_reduce.FINDINGS_SOURCE_ID])))

    result = _run(fake_post)

    assert len(reduce_calls) == 2, f"expected exactly two synthesis calls, got {len(reduce_calls)}"
    second = _user_content(reduce_calls[1])
    assert "at most 8" in second.lower(), (
        "the retry after a length stop must ask for at most 8 findings per section; "
        "it did not mention a per-section limit"
    )
    assert "section" in second.lower()
    assert "YOUR PREVIOUS ANSWER WAS REJECTED BY LOCAL VALIDATION" not in second, (
        "a length cut was fed back as a validation rejection, which re-sends the "
        "same request with MORE prompt against the same ceiling"
    )
    assert result["map_reduce"]["synthesized"] is True
    assert result["map_reduce"].get("synthesis_retry") == "shorter", (
        "the manifest must record that the synthesis retried shorter; got "
        f"{result['map_reduce'].get('synthesis_retry')!r}"
    )


def test_two_length_stops_raise_with_the_reason():
    """Cut twice -> the RuntimeError says LENGTH, not 'invalid JSON'.

    'Unterminated string starting at: line 1 column 14709' sent the lead
    looking for a malformed model; the manifest must name the stop reason.
    """
    reduce_calls = []

    def fake_post(url, **kwargs):
        if not _is_reduce_call(kwargs):
            return _map_reply(kwargs)
        reduce_calls.append(kwargs)
        return _chat(CUT_JSON, finish_reason="length")

    result = _run(fake_post)
    block = result["map_reduce"]

    assert block["synthesized"] is False
    assert "length" in block["synthesis_error"].lower(), (
        f"the error must name the stop reason; got {block['synthesis_error']!r}"
    )
    assert block.get("synthesis_stop_reason") == "length", (
        f"synthesis_stop_reason must be 'length'; got {block.get('synthesis_stop_reason')!r}"
    )


def test_an_ollama_done_reason_is_a_length_stop_too():
    """Ollama reports ``done_reason`` at the top level, not on the choice."""
    import ai_summary

    def fake_post(url, **kwargs):
        return _chat(CUT_JSON, finish_reason=None, done_reason="length")

    with _settings(ai_local_endpoint_url=ENDPOINT):
        with pytest.raises(RuntimeError) as ctx:
            ai_summary.request_ai_summary(
                provider="local",
                model="gemma3:12b",
                api_key="",
                evidence=_package([("a.one", ["x"])]),
                post=fake_post,
            )
    assert "length" in str(ctx.value).lower(), (
        f"an Ollama length stop was reported as bad JSON: {ctx.value}"
    )


def test_a_map_slice_that_stops_for_length_is_retried_shorter_and_recorded():
    """The map path gets the same detection (packet item 2, map half).

    The block's SHAPE for this is the builder's choice, so this asserts only
    that the slice was read rather than failed and that the retry is stated
    somewhere in the manifest block - not where.
    """
    seen = {"a.one": 0}

    def fake_post(url, **kwargs):
        if _is_reduce_call(kwargs):
            return _chat(json.dumps(_summary("synthesized", [map_reduce.FINDINGS_SOURCE_ID])))
        content = _user_content(kwargs)
        if "a.one" in content:
            seen["a.one"] += 1
            if seen["a.one"] == 1:
                return _chat(CUT_JSON, finish_reason="length")
        return _map_reply(kwargs)

    result = _run(fake_post)
    block = result["map_reduce"]

    assert block["slices_read"] == 2 and block["slices_failed"] == []
    assert "shorter" in json.dumps(block), (
        "a map slice that stopped for length and then answered shorter left no "
        f"trace in the manifest block: {json.dumps(block)}"
    )


# ------------------------------------------------------------------ item 3


def test_the_manifest_always_carries_a_stop_reason_key():
    """Absent -> "" so a reader can tell 'not measured' from 'stopped'."""

    def fake_post(url, **kwargs):
        if _is_reduce_call(kwargs):
            return _chat(json.dumps(_summary("synthesized", [map_reduce.FINDINGS_SOURCE_ID])))
        return _map_reply(kwargs)

    block = _run(fake_post)["map_reduce"]
    assert block["synthesized"] is True
    assert "synthesis_stop_reason" in block, (
        "the key must always be present; a reader cannot tell an unmeasured "
        "stop from a clean one otherwise"
    )
    assert block["synthesis_stop_reason"] == ""


# ------------------------------------------------------------------ item 4


def test_currently_shorted_is_a_position_claim():
    """'APPS is currently shorted' reached ai_morning_brief.txt on 2026-09-05.

    The alternation is wrapped in ``\\b(...)\\b``, so 'currently short' followed
    by 'ed' has no trailing word boundary and the whole pattern misses.
    """
    from ai_summary import states_a_position

    assert states_a_position("APPS is currently shorted")
    assert states_a_position("APPS is currently longed")
    # Unchanged behaviour on either side of the one-word fix.
    assert states_a_position("BULL is currently long")
    assert states_a_position("we are currently short BULL")
    assert not states_a_position("APPS is currently shorting sellers of volatility")
    assert not states_a_position("BULL is on the longs watchlist")


# ------------------------------------------------------------------ item 6 (guard)


def test_unsynthesized_fallback_unchanged():
    """The last-resort path stays exactly as it was.

    A retry that swallows the failure would publish a document that looks
    synthesized. The executive line must still open UNSYNTHESIZED and the
    per-section cap must still be MAX_UNSYNTHESIZED_ROWS_PER_SECTION.
    """

    def fake_post(url, **kwargs):
        if _is_reduce_call(kwargs):
            return _chat(CUT_JSON, finish_reason="length")
        return _map_reply(kwargs)

    result = _run(fake_post)

    assert result["map_reduce"]["synthesized"] is False
    assert result["map_reduce"]["slices_read"] == 2
    assert result["summary"]["executive_summary"].startswith("UNSYNTHESIZED")
    statements = [row["statement"] for row in result["summary"]["what_is_working"]]
    assert "a finding from a.one" in statements and "a finding from b.two" in statements

    findings = {
        "what_is_working": [
            {"statement": f"f{i}", "evidence_refs": ["a.one"], "confidence": "low"}
            for i in range(30)
        ]
    }
    out = map_reduce.unsynthesized_summary(findings, read=30, planned=30)
    assert len(out["what_is_working"]) == map_reduce.MAX_UNSYNTHESIZED_ROWS_PER_SECTION + 1
    assert "further finding(s)" in out["what_is_working"][-1]["statement"]
