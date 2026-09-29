"""The night AI falls back to the local model when the RTX 5080 host is lost.

`run_ai_jobs.ps1` hands the job an endpoint override for the ssh tunnel. If the
tunnel or the host dies mid-run, the job must finish on the local endpoint, and
it must leave the night's dead-flag so later firings skip the host.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ai_summary  # noqa: E402
from test_local_ai_provider import _chat_response, _daily_overrides, _valid_summary  # noqa: E402

LOCAL = "http://127.0.0.1:11434/v1"
REMOTE = "http://127.0.0.1:11435/v1"


class _Up:
    status_code = 200


@pytest.fixture
def remote(monkeypatch, tmp_path):
    monkeypatch.setitem(ai_summary._remote_state, "dead", False)
    monkeypatch.setitem(ai_summary._remote_state, "checked_at", None)
    flag = tmp_path / "gpu_host_dead-20260928.flag"
    monkeypatch.setenv(ai_summary.LOCAL_ENDPOINT_OVERRIDE_ENV, REMOTE)
    monkeypatch.setenv(ai_summary.REMOTE_DEAD_FLAG_ENV, str(flag))
    monkeypatch.setattr(
        ai_summary, "get_local_setting",
        lambda key, default=None: LOCAL if key == ai_summary.LOCAL_ENDPOINT_SETTING_KEY else default,
    )
    return flag


def test_a_healthy_remote_is_used(remote):
    with mock.patch.object(ai_summary.requests, "get", return_value=_Up()) as get:
        assert ai_summary.local_endpoint_url() == REMOTE
        assert ai_summary.local_endpoint_url() == REMOTE
    # The passing check is trusted for a while rather than asked per call.
    assert get.call_count == 1
    assert get.call_args.args[0] == "http://127.0.0.1:11435/api/version"


def test_a_failed_health_check_goes_local_for_the_rest_of_the_run(remote):
    with mock.patch.object(ai_summary.requests, "get", side_effect=ConnectionError("refused")):
        assert ai_summary.local_endpoint_url() == LOCAL
    assert remote.exists()
    with mock.patch.object(ai_summary.requests, "get", return_value=_Up()):
        assert ai_summary.local_endpoint_url() == LOCAL


def test_an_existing_dead_flag_skips_the_remote(remote):
    remote.write_text("earlier firing", encoding="utf-8")
    with mock.patch.object(ai_summary.requests, "get", return_value=_Up()) as get:
        assert ai_summary.local_endpoint_url() == LOCAL
    get.assert_not_called()


def test_no_override_means_the_saved_endpoint(remote, monkeypatch):
    monkeypatch.delenv(ai_summary.LOCAL_ENDPOINT_OVERRIDE_ENV)
    with mock.patch.object(ai_summary.requests, "get") as get:
        assert ai_summary.local_endpoint_url() == LOCAL
    get.assert_not_called()


def test_the_override_never_switches_the_provider_on(remote, monkeypatch):
    monkeypatch.setattr(ai_summary, "get_local_setting", lambda key, default=None: default)
    assert ai_summary.local_endpoint_url() == ""


def test_a_request_lost_mid_run_is_resent_to_the_local_endpoint(remote):
    with tempfile.TemporaryDirectory() as raw:
        evidence = ai_summary.build_evidence_package(
            ["daily_report"], source_overrides=_daily_overrides(Path(raw))
        )
    summary_text = json.dumps(_valid_summary("daily.auto_report"))
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        if url.startswith(REMOTE):
            raise ConnectionError("tunnel closed")
        return _chat_response(summary_text)

    with mock.patch.object(ai_summary.requests, "get", return_value=_Up()):
        result = ai_summary.request_ai_summary(
            provider="local", model="gemma3:12b", api_key="", evidence=evidence, post=fake_post,
        )

    assert result["status"] == "validated"
    assert calls == [f"{REMOTE}/chat/completions", f"{LOCAL}/chat/completions"]
    assert remote.exists()


def test_a_local_failure_is_still_a_clean_unreachable_error(remote, monkeypatch):
    monkeypatch.delenv(ai_summary.LOCAL_ENDPOINT_OVERRIDE_ENV)
    with tempfile.TemporaryDirectory() as raw:
        evidence = ai_summary.build_evidence_package(
            ["daily_report"], source_overrides=_daily_overrides(Path(raw))
        )

    def refused(url, **kwargs):
        raise ConnectionError("refused")

    with pytest.raises(ai_summary.LocalEndpointUnreachable):
        ai_summary.request_ai_summary(
            provider="local", model="gemma3:12b", api_key="", evidence=evidence, post=refused,
        )
