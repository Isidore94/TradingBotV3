"""Red regressions for the 2026-09-17 overnight repair packet.

These tests use the actual import and local-provider paths.  The subprocess
sets the data root before importing application modules, so it cannot touch a
live store.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
LOCAL_ENDPOINT = "http://127.0.0.1:11434/v1"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class _Response:
    def __init__(self, payload: dict, *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self.payload


def _chat_response(text: str, *, status_code: int = 200) -> _Response:
    return _Response(
        {
            "id": "chatcmpl-overnight-repair",
            "choices": [{"message": {"role": "assistant", "content": text}}],
        },
        status_code=status_code,
    )


def _grammar_failure() -> _Response:
    return _Response(
        {"error": {"message": "failed to parse grammar while initializing schema"}},
        status_code=400,
    )


def _enrichment_response(*, summary: str = "Measured advisory summary.") -> dict:
    return {
        "summary": summary,
        "tags": ["AVWAP reclaim"],
        "confidence": "medium",
        "sources": ["trade:T-overnight"],
        "unknowns": [],
    }


def _local_enrichment_request(ai_summary, post):
    from ai_jobs.enrichment import ENRICHMENT_JSON_SCHEMA

    return ai_summary.request_ai_summary(
        provider="local",
        model="gemma3:12b",
        api_key="",
        evidence={"package_id": "overnight-repair", "evidence_hash": "pinned"},
        schema=ENRICHMENT_JSON_SCHEMA,
        schema_name="ai_trade_enrichment",
        prompt_version="ai_trade_enrichment_v1",
        post=post,
    )


def _local_settings(ai_summary):
    return mock.patch.object(
        ai_summary,
        "get_local_setting",
        lambda key, default=None: {"ai_local_endpoint_url": LOCAL_ENDPOINT}.get(key, default),
    )


def test_theta_tracker_cold_import_keeps_runner_recorder_monkeypatch_seam(tmp_path):
    """A fresh interpreter must import the tracker and retain the runner seam."""
    child = tmp_path / "cold_theta_import.py"
    child.write_text(
        """
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

scratch = Path(sys.argv[1])
scripts = Path(sys.argv[2])
os.environ['TRADINGBOTV3_DATA_DIR'] = str(scratch / 'home')
os.environ['LOCALAPPDATA'] = str(scratch / 'localappdata')
os.environ['TRADINGBOT_DIAGNOSTICS_DIR'] = str(scratch / 'diagnostics')
os.environ['TRADINGBOT_DISABLE_BACKGROUND_MAINTENANCE'] = '1'
sys.path.insert(0, str(scripts))

import project_paths
if 'TradingBotData' in str(project_paths.DATA_DIR):
    raise SystemExit(f'unsafe data directory: {project_paths.DATA_DIR}')
import theta_pick_tracker
from master_avwap_lib import runner

real_recorder = runner.record_theta_picks
assert callable(real_recorder)
assert real_recorder([], [], '2026-09-16', datetime.now(timezone.utc), path=scratch / 'theta.jsonl') == 0
seen = []
runner.record_theta_picks = lambda *args, **kwargs: seen.append((args, kwargs))
runner.record_theta_picks([], [], '2026-09-16', datetime.now(timezone.utc))
assert len(seen) == 1
print('THETA-COLD-IMPORT-OK')
""",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(child), str(tmp_path), str(SCRIPTS)],
        cwd=str(SCRIPTS),
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "THETA-COLD-IMPORT-OK" in completed.stdout


def test_local_enrichment_retries_only_grammar_400_with_json_object_and_full_contract():
    """The grammar workaround is one JSON-object retry, not a weaker contract."""
    import ai_summary

    requests: list[dict] = []
    responses = [_grammar_failure(), _chat_response(json.dumps(_enrichment_response()))]

    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return responses[len(requests) - 1]

    with _local_settings(ai_summary):
        result = _local_enrichment_request(ai_summary, post)

    assert result["status"] == "validated"
    assert result["summary"] == _enrichment_response()
    assert len(requests) == 2
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert requests[1]["response_format"] == {"type": "json_object"}


def test_local_enrichment_grammar_fallback_still_rejects_an_incomplete_contract():
    """JSON-object fallback must validate the original closed enrichment schema."""
    import ai_summary

    requests: list[dict] = []
    responses = [_grammar_failure(), _chat_response(json.dumps({"summary": "only this key"}))]

    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return responses[len(requests) - 1]

    with _local_settings(ai_summary):
        with pytest.raises(RuntimeError, match="missing required field"):
            _local_enrichment_request(ai_summary, post)

    assert len(requests) == 2
    assert requests[1]["response_format"] == {"type": "json_object"}


def test_local_enrichment_validation_enforces_its_2000_character_summary_limit():
    """The fallback's local validation must enforce the contract's maxLength."""
    import ai_summary

    requests: list[dict] = []
    responses = [
        _grammar_failure(),
        _chat_response(json.dumps(_enrichment_response(summary="x" * 2001))),
    ]

    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return responses[len(requests) - 1]

    with _local_settings(ai_summary):
        with pytest.raises(RuntimeError, match="2000"):
            _local_enrichment_request(ai_summary, post)

    assert len(requests) == 2
    assert requests[1]["response_format"] == {"type": "json_object"}


def test_local_enrichment_does_not_retry_a_non_grammar_http_400():
    """A model or request 400 remains one failed request, never a broad retry."""
    import ai_summary

    requests: list[dict] = []

    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return _Response({"error": {"message": "model does not exist"}}, status_code=400)

    with _local_settings(ai_summary):
        with pytest.raises(RuntimeError, match="local request failed \\(400\\)"):
            _local_enrichment_request(ai_summary, post)

    assert len(requests) == 1
    assert requests[0]["response_format"]["type"] == "json_schema"
