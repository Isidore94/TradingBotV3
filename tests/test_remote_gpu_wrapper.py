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


def test_a_host_problem_never_refuses_the_run():
    body = _remote_function()
    assert "exit " not in body
    assert "throw" not in body
    assert body.count("will run degraded") >= 3


def test_the_tunnel_binds_loopback_only_and_targets_the_hosts_ollama_port():
    body = _remote_function()
    assert '"127.0.0.1:${TunnelPort}:127.0.0.1:11434"' in body


def test_the_host_start_script_ships_with_lf_line_endings():
    script = SCRIPTS_DIR / "remote_gpu" / "ollama_up.sh"
    data = script.read_bytes()
    assert data.startswith(b"#!/usr/bin/env bash")
    assert b"\r" not in data
    assert b"127.0.0.1:11434" in data
