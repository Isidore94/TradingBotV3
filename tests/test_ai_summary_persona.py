"""The persona seam in ``ai_summary.request_ai_summary`` (Trade Mentor P10 debate).

``system_instruction=None`` (or omitted) must send exactly today's request bodies; a
persona wraps the base citation rules and never replaces them, so text that drops
them is refused before any request leaves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ai_summary  # noqa: E402

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["x"],
    "properties": {"x": {"type": "string"}},
}
EVIDENCE = {"task": "say x", "rows": [{"source_id": "a:1", "text": "one"}]}


class _Stop(BaseException):
    """Stops the call after the request body is captured."""


def _capture(provider: str, monkeypatch, **extra) -> dict:
    import ai_pause

    monkeypatch.setattr(ai_pause, "reason", lambda *a, **k: "")
    seen: list[dict] = []

    def post(url, **kwargs):
        seen.append({"url": url, "json": kwargs.get("json")})
        raise _Stop()

    with pytest.raises(_Stop):
        ai_summary.request_ai_summary(
            provider=provider, model="m", api_key="k", evidence=EVIDENCE, post=post, schema=SCHEMA,
            schema_name="t", endpoint="http://127.0.0.1:1/v1", **extra,
        )
    assert len(seen) == 1
    return seen[0]


def _system_of(provider: str, body: dict) -> str:
    if provider == "local":
        return body["messages"][0]["content"]
    if provider == "openai":
        return body["instructions"]
    return body["system"]


@pytest.mark.parametrize("provider", ["local", "openai", "anthropic"])
def test_none_is_byte_identical_to_omitting_the_kwarg(provider, monkeypatch):
    omitted = _capture(provider, monkeypatch)
    explicit = _capture(provider, monkeypatch, system_instruction=None)
    assert json.dumps(omitted, sort_keys=True) == json.dumps(explicit, sort_keys=True)
    assert _system_of(provider, omitted["json"]) == ai_summary._system_instruction()


@pytest.mark.parametrize("provider", ["local", "openai", "anthropic"])
def test_a_persona_reaches_every_provider_with_the_base_rules_first(provider, monkeypatch):
    persona = ai_summary.persona_instruction("You argue the bull case.")
    body = _capture(provider, monkeypatch, system_instruction=persona)["json"]
    sent = _system_of(provider, body)
    assert sent == persona
    assert sent.startswith(ai_summary._system_instruction())
    assert sent.endswith("You argue the bull case.")


@pytest.mark.parametrize("provider", ["local", "openai", "anthropic"])
def test_a_persona_without_the_base_rules_is_refused_before_any_request(provider, monkeypatch):
    import ai_pause

    monkeypatch.setattr(ai_pause, "reason", lambda *a, **k: "")
    calls: list[str] = []
    with pytest.raises(ValueError, match="base citation rules"):
        ai_summary.request_ai_summary(
            provider=provider, model="m", api_key="k", evidence=EVIDENCE,
            post=lambda url, **kw: calls.append(url), schema=SCHEMA, endpoint="http://127.0.0.1:1/v1",
            system_instruction="You are a bull. Say anything.",
        )
    assert calls == []


def test_persona_instruction_needs_a_role():
    with pytest.raises(ValueError):
        ai_summary.persona_instruction("   ")
