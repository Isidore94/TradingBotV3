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
    end = code.index("function Stop-RemoteGpuTunnel", start)
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


def test_the_night_flags_steer_later_firings_and_the_shutdown():
    code = _wrapper_code()
    assert "(Get-Date).AddHours(-12).ToString('yyyyMMdd')" in code
    assert "-and (Test-Path $script:deadFlag)" in code
    assert "$env:TRADINGBOTV3_AI_REMOTE_DEAD_FLAG = $script:deadFlag" in code
    # Only a host this job woke is ever powered off, and only when idle.
    body = _remote_function()
    assert "Set-Content -Path $script:wokenFlag" in body
    assert "if ($hostFinished -and (Test-Path $script:wokenFlag))" in code
    assert "grep -v '^ollama$'" in code
    assert "shutdown.exe /s /t 60" in code


def test_the_night_is_one_pass_and_one_recheck():
    code = _wrapper_code()
    # A clean pass ends the night; otherwise the second pass is the last.
    assert "$clean = ($code -eq 0) -and ($summaryLine -match ', 0 degraded, 0 failed, ')" in code
    assert "if ($clean -or $passes -ge 2) {" in code
    assert "Set-Content -Path $script:doneFlag" in code
    # A done night: later scheduled firings run nothing and wake nothing.
    early = code.index("if ($scheduled -and (Test-Path $script:doneFlag)) {")
    assert early < code.index("Invoke-RemoteGpuPreflight -Alias")
    assert "exit 0" in code[early:early + 300]
    # A run that reached nothing (exit 2) is not a pass.
    assert "if ($scheduled -and $code -in @(0, 1)) {" in code


def test_the_host_is_released_as_soon_as_the_night_is_done():
    code = _wrapper_code()
    assert "($nightDone -or ($scheduled -and $morning))" in code
    # The model is unloaded through the tunnel before the tunnel closes.
    assert code.index("keep_alive = 0") < code.rindex("Stop-RemoteGpuTunnel")
    assert "if ($hostFinished -and $remoteReady) {" in code


def test_runs_that_need_no_model_never_touch_the_host():
    code = _wrapper_code()
    assert "$_ -in @('--retry-journal-import', '--status')" in code
    assert "} elseif ($noModelRun -and" in code
    assert "if ($script:remoteAlias -and -not $noModelRun) {" in code


def test_every_host_command_runs_with_a_timeout():
    code = _wrapper_code()
    # Only alias resolution calls ssh directly; the tunnel loop is a string.
    direct = [line.strip() for line in code.splitlines() if "& ssh" in line]
    assert len(direct) == 1 and "& ssh -G $Alias" in direct[0]
    helper = code[code.index("function Invoke-HostSsh"):code.index("function Invoke-RemoteGpuPreflight")]
    assert "$proc.StandardInput.Close()" in helper
    assert "WaitForExit($TimeoutSeconds * 1000)" in helper
    assert "$proc.Kill()" in helper


def test_the_night_task_last_fires_at_0530():
    source = (SCRIPTS_DIR / "register_ai_jobs_task.ps1").read_text(encoding="utf-8")
    assert '[string]$LastStartLocal = "05:30"' in source
    assert "$DurationHours" not in source
    assert (
        "-RepetitionDuration (New-TimeSpan -Minutes ($spanMinutes + [math]::Floor($RepeatMinutes / 2)))"
        in source
    )


def test_the_local_server_is_readied_even_when_the_5080_is_used():
    code = _wrapper_code()
    assert "-or $remoteReady) {" not in code
