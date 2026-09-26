"""P8b B1: the night is measurable - tokens per slot, goal coverage, Health rows."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ET = ZoneInfo("America/New_York")
NIGHT = datetime(2026, 8, 12, 2, 0, tzinfo=ET)
SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "string"}},
    "required": ["a"],
    "additionalProperties": False,
}


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _Reply:
    status_code = 200
    text = ""

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _fake_post(usages: list[dict]):
    """A local chat endpoint answering valid JSON with the given usage blocks in turn."""
    queue = list(usages)

    def post(url, **kwargs):
        usage = queue.pop(0)
        body = {
            "id": "r1",
            "choices": [{"message": {"content": json.dumps({"a": "x"})}, "finish_reason": "stop"}],
        }
        if usage:
            body["usage"] = usage
        return _Reply(body)

    return post


def _ask(post):
    import ai_summary

    return ai_summary.request_ai_summary(
        provider="local",
        model="m",
        api_key="",
        evidence={"package_id": "p", "evidence_hash": "h", "sources": []},
        post=post,
        schema=SCHEMA,
        schema_name="t",
    )


def _night(monkeypatch):
    import ai_summary
    from ai_jobs import store, window

    monkeypatch.setattr(store, "store_available", lambda: (True, "ready"))
    monkeypatch.setattr(window, "launch_allowed", lambda *a, **k: (True, "window open"))
    monkeypatch.setattr(window, "market_session_block", lambda *a, **k: "")
    monkeypatch.setattr(ai_summary, "local_endpoint_url", lambda: "http://127.0.0.1:11434/v1")


# ---------------------------------------------------------------------------
# 1. Tokens per slot
# ---------------------------------------------------------------------------


def test_every_model_calling_slot_row_carries_its_summed_tokens(tmp_path, monkeypatch):
    from ai_jobs import runner

    _night(monkeypatch)
    post = _fake_post([
        {"prompt_tokens": 100, "completion_tokens": 10},
        {"prompt_tokens": 50, "completion_tokens": 5},
        {"prompt_tokens": 7, "completion_tokens": 1},
    ])

    def two_calls(**kwargs):
        _ask(post)
        _ask(post)
        return {"status": "ok"}

    def one_call_then_crash(**kwargs):
        _ask(post)
        raise RuntimeError("boom")

    led = tmp_path / "ledger.jsonl"
    slots = [
        runner.JobSlot(name="a", run=two_calls, goal="coaching"),
        runner.JobSlot(name="b", run=lambda **k: {"status": "ok"}, goal="ops"),
        runner.JobSlot(name="c", run=one_call_then_crash, goal="journal"),
    ]
    runner.run_slots(slots, now=NIGHT, ledger_path=led)
    tokens = {row["job"]: row["tokens"] for row in _rows(led)}
    assert tokens["a"] == {"prompt_tokens": 150, "completion_tokens": 15, "calls": 2}
    assert tokens["b"] == {}
    assert tokens["c"] == {"prompt_tokens": 7, "completion_tokens": 1, "calls": 1}


def test_a_slots_own_richer_tokens_dict_is_merged_not_clobbered(tmp_path, monkeypatch):
    from ai_jobs import runner

    _night(monkeypatch)
    post = _fake_post([
        {"prompt_tokens": 40, "completion_tokens": 4},
        {},
    ])

    def briefs(**kwargs):
        _ask(post)
        _ask(post)
        return {"status": "ok", "tokens": {"ticker_calls": 2, "prompt_tokens": 40}}

    led = tmp_path / "ledger.jsonl"
    runner.run_slots(
        [runner.JobSlot(name="ticker_briefs", run=briefs, goal="market_read")],
        now=NIGHT,
        ledger_path=led,
    )
    assert _rows(led)[0]["tokens"] == {
        "ticker_calls": 2,
        "prompt_tokens": 40,
        "completion_tokens": 4,
        "calls": 2,
    }
