"""Pins for the remote-GPU branch of `scripts/run_ai_jobs.ps1`.

The wrapper can send the night AI's inference to the RTX 5080 host over an ssh
tunnel. It is off unless `ai_remote_gpu_ssh_alias` is set, and it may never turn
a host problem into a refused run: the deterministic jobs do not need a model.
"""

from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _wrapper_code() -> str:
    source = (SCRIPTS_DIR / "run_ai_jobs.ps1").read_text(encoding="utf-8")
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )


def _remote_function() -> str:
    code = _wrapper_code()
    start = code.index("function Invoke-RemoteGpuPreflight")
    end = code.index("\ntry {", start)
    return code[start:end]


def test_remote_gpu_is_off_unless_the_alias_setting_is_present():
    code = _wrapper_code()
    assert "ai_remote_gpu_ssh_alias" in code
    assert "-not [string]::IsNullOrWhiteSpace($script:remoteAlias)" in code


def test_a_host_problem_falls_back_to_the_local_server():
    body = _remote_function()
    assert "exit " not in body
    assert "throw" not in body
    assert body.count("using the local server") >= 5
    code = _wrapper_code()
    # The override is set only after a passing preflight.
    assert "if ($remoteReady) {\n            $env:TRADINGBOTV3_AI_ENDPOINT_OVERRIDE" in code


def test_log_lines_cannot_pass_for_a_ready_verdict():
    code = _wrapper_code()
    assert "$remoteReady = Invoke-RemoteGpuPreflight" not in code
    assert "($preflight[-1] -is [bool]) -and $preflight[-1]" in code


def test_the_saved_endpoint_setting_is_never_written():
    code = _wrapper_code()
    assert "Set-Content -Path $settingsPath" not in code
    assert "ConvertTo-Json | Set-Content" not in code


def test_the_tunnel_binds_loopback_only_and_targets_the_hosts_ollama_port():
    body = _remote_function()
    assert "-L 127.0.0.1:${TunnelPort}:127.0.0.1:11434" in body
    assert "while (`$true)" in body


def test_the_model_tag_is_checked_and_warmed_before_the_handover():
    body = _remote_function()
    assert "/api/tags" in body
    assert "/api/generate" in body


def test_the_host_start_script_ships_with_lf_line_endings():
    script = SCRIPTS_DIR / "remote_gpu" / "ollama_up.sh"
    data = script.read_bytes()
    assert data.startswith(b"#!/usr/bin/env bash")
    assert b"\r" not in data
    assert b"127.0.0.1:11434" in data


def test_the_override_redirects_only_an_enabled_provider(monkeypatch):
    import ai_summary

    monkeypatch.setenv(ai_summary.LOCAL_ENDPOINT_OVERRIDE_ENV, "http://127.0.0.1:11435/v1/")
    monkeypatch.setattr(ai_summary, "get_local_setting", lambda key, default=None: "http://127.0.0.1:11434/v1")
    assert ai_summary.local_endpoint_url() == "http://127.0.0.1:11435/v1"

    monkeypatch.setattr(ai_summary, "get_local_setting", lambda key, default=None: "")
    assert ai_summary.local_endpoint_url() == ""


def test_without_the_override_the_setting_wins(monkeypatch):
    import ai_summary

    monkeypatch.delenv(ai_summary.LOCAL_ENDPOINT_OVERRIDE_ENV, raising=False)
    monkeypatch.setattr(ai_summary, "get_local_setting", lambda key, default=None: "http://127.0.0.1:11434/v1/")
    assert ai_summary.local_endpoint_url() == "http://127.0.0.1:11434/v1"
