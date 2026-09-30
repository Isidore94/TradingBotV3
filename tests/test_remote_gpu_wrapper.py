"""Pins for the remote-GPU branch of `scripts/run_ai_jobs.ps1`.

The wrapper can send the night AI's inference to the RTX 5080 host over an ssh
tunnel. It is off unless `ai_remote_gpu_ssh_alias` is set, and it may never turn
a host problem into a refused run: the deterministic jobs do not need a model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


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


# ---------------------------------------------------------------- Pause AI (2026-09-30)
def test_pause_ai_is_read_first_and_leaves_the_host_alone():
    code = _wrapper_code()
    assert "$aiPausedUntil = Get-AiPausedUntil -Path $settingsPath" in code
    paused = code.index("if ($aiPausedUntil) {")
    # The pause branch comes before every remote-GPU branch, so the preflight never runs.
    assert paused < code.index("Invoke-RemoteGpuPreflight -Alias")
    branch = code[paused:code.index("} elseif", paused)]
    assert "$noModelRun = $true" in branch
    assert 'Write-Log "AI paused until $aiPausedUntil; remote GPU untouched"' in branch
    assert "TRADINGBOTV3_AI_ENDPOINT_OVERRIDE" not in branch
    # $noModelRun is what keeps the host-log mirror and the power-off from running.
    assert "$hostFinished = $script:remoteAlias -and -not $noModelRun -and" in code


def _powershell() -> str:
    import shutil

    powershell = shutil.which("powershell.exe")
    if not powershell:
        import pytest

        pytest.skip("needs Windows PowerShell")
    return powershell


#: A stand-in for python.exe and ssh.exe: records every call; as python it may rewrite the
#: settings file (a pause set mid-run) and prints a clean night summary.
_FAKE_SOURCE = r"""
using System; using System.Diagnostics; using System.IO;
public static class FakeTool { public static int Main(string[] args) {
    string name = Path.GetFileName(Process.GetCurrentProcess().MainModule.FileName).ToLowerInvariant();
    string calls = Environment.GetEnvironmentVariable("FAKE_CALLS");
    if (!String.IsNullOrEmpty(calls)) File.AppendAllText(calls, name + " " + String.Join(" ", args) + Environment.NewLine);
    if (name.StartsWith("python")) {
        string settings = Environment.GetEnvironmentVariable("FAKE_CHILD_SETTINGS");
        if (!String.IsNullOrEmpty(settings)) File.WriteAllText(Environment.GetEnvironmentVariable("FAKE_SETTINGS_FILE"), settings);
        Console.WriteLine("AI jobs for session 2026-09-30: 3 ok, 0 degraded, 0 failed, 0 skipped");
    }
    return 0;
} }
"""


def _fake_tool(tmp_path: Path) -> Path:
    import subprocess

    source = tmp_path / "fake_tool.cs"
    source.write_text(_FAKE_SOURCE, encoding="utf-8")
    exe = tmp_path / "fake_tool.exe"
    subprocess.run(
        [_powershell(), "-NoProfile", "-Command",
         f"Add-Type -TypeDefinition (Get-Content '{source}' -Raw) -OutputAssembly '{exe}' -OutputType ConsoleApplication"],
        capture_output=True, text=True, timeout=120, check=True,
    )
    return exe


def _run_wrapper(tmp_path: Path, settings: dict, *, child_settings: dict | None = None,
                 fake_ssh: bool = False, woken: bool = False) -> str:
    """Run the real wrapper in a scratch tree; returns its log. A fake interpreter stands in
    for the venv Python, and a scratch HOME/LOCALAPPDATA keeps it off the live machine."""
    import json
    import os
    import shutil
    import subprocess
    from datetime import datetime, timedelta

    powershell = _powershell()
    root = tmp_path / "repo"
    (root / "scripts" / "remote_gpu").mkdir(parents=True)
    shutil.copy(SCRIPTS_DIR / "run_ai_jobs.ps1", root / "scripts" / "run_ai_jobs.ps1")
    shutil.copy(SCRIPTS_DIR / "remote_gpu" / "ollama_up.sh", root / "scripts" / "remote_gpu" / "ollama_up.sh")
    (root / "scripts" / "run_ai_jobs.py").write_text("", encoding="utf-8")
    (root / ".venv" / "Scripts").mkdir(parents=True)
    fake = _fake_tool(tmp_path)
    shutil.copy(fake, root / ".venv" / "Scripts" / "python.exe")
    appdata = tmp_path / "appdata"
    (appdata / "TradingBotV3").mkdir(parents=True)
    settings_file = appdata / "TradingBotV3" / "local_settings.json"
    settings_file.write_text(json.dumps(settings), encoding="utf-8")
    if woken:
        night = (datetime.now() - timedelta(hours=12)).strftime("%Y%m%d")
        (appdata / "TradingBotV3" / f"gpu_host_woken-{night}.flag").write_text("x", encoding="ascii")
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "LOCALAPPDATA": str(appdata), "USERPROFILE": str(home),
           "HOMEDRIVE": str(home)[:2], "HOMEPATH": str(home)[2:],
           "FAKE_CALLS": str(tmp_path / "calls.txt"), "FAKE_SETTINGS_FILE": str(settings_file),
           "FAKE_CHILD_SETTINGS": json.dumps(child_settings) if child_settings is not None else ""}
    if fake_ssh:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        shutil.copy(fake, bin_dir / "ssh.exe")
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    subprocess.run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root / "scripts" / "run_ai_jobs.ps1")],
        env=env, capture_output=True, text=True, timeout=180,
    )
    logs = sorted((appdata / "TradingBotV3" / "logs").glob("ai_jobs-*.log"))
    assert logs, "the wrapper wrote no log"
    return logs[-1].read_text(encoding="utf-8-sig")


#: An alias that resolves nowhere and a TEST-NET endpoint: nothing real can be reached.
_SETTINGS = {
    "ai_local_endpoint_url": "http://192.0.2.1:11434/v1",
    "ai_remote_gpu_ssh_alias": "pause-ai-test.invalid",
}


def test_a_paused_night_run_touches_no_remote_gpu(tmp_path):
    log = _run_wrapper(tmp_path, {**_SETTINGS, "ai_paused_until": "2999-01-01T00:00:00-08:00"})
    assert "AI paused until 2999-01-01T00:00:00-08:00; remote GPU untouched" in log
    assert "remote GPU:" not in log, "no preflight, no mirror, no power-off while paused"
    assert not list((tmp_path / "appdata" / "TradingBotV3").glob("gpu_host_*.flag"))


def test_an_expired_pause_runs_the_preflight_as_before(tmp_path):
    log = _run_wrapper(tmp_path, {**_SETTINGS, "ai_paused_until": "2020-01-01T00:00:00-08:00"})
    assert "AI paused" not in log
    assert "remote GPU:" in log


def test_a_pause_set_while_the_jobs_run_never_powers_the_host_off(tmp_path):
    """Reviewer's repro: the host was woken by this job, the night comes out clean, and the
    trader pauses AI (to game) while the child runs. The host must stay on."""
    log = _run_wrapper(
        tmp_path, _SETTINGS, fake_ssh=True, woken=True,
        child_settings={**_SETTINGS, "ai_paused_until": "2999-01-01T00:00:00-08:00"},
    )
    calls = (tmp_path / "calls.txt").read_text(encoding="utf-8")
    assert "night done after pass 1" in log, "the setup reached the power-off decision"
    assert "AI paused mid-run; host left on" in log
    assert "shutdown" not in log
    assert "shutdown.exe" not in calls and "tmux ls" not in calls


def test_the_mid_run_pause_unloads_the_nights_model_before_leaving_the_host_on():
    code = _wrapper_code()
    start = code.index("$midRunPause = (-not $aiPausedUntil) -and [bool](Get-AiPausedUntil -Path $settingsPath)")
    block = code[start:code.index("$hostFinished =", start)]
    assert "if ($remoteReady) {" in block and "keep_alive = 0" in block
    assert block.index("keep_alive = 0") < block.index('Write-Log "AI paused mid-run; host left on"')
    assert "-and -not $midRunPause -and" in code[code.index("$hostFinished ="):]
    # Re-read after the child exits, before the unload and power-off decision.
    assert code.index("Start-Process -FilePath $python") < start < code.index("if ($hostFinished -and $remoteReady) {")


# ---------------------------------------------------------------- one parse rule, both sides
@pytest.mark.parametrize(
    "raw, paused",
    [
        ("2999-01-01T00:00:00-08:00", True),
        ("2999-01-01T08:00:00Z", True),
        ("2020-01-01T00:00:00-08:00", False),
        ("2999-01-01T00:00:00", False),  # naive: rejected on both sides
        ("junk", False),
        (None, False),  # missing
    ],
)
def test_the_wrapper_and_python_read_the_pause_the_same_way(tmp_path, monkeypatch, raw, paused):
    import json
    import subprocess

    import ai_pause
    import project_paths

    settings_file = tmp_path / "local_settings.json"
    settings_file.write_text(json.dumps({} if raw is None else {"ai_paused_until": raw}), encoding="utf-8")
    monkeypatch.setattr(project_paths, "LOCAL_SETTINGS_FILE", settings_file)
    project_paths.invalidate_local_settings_cache()
    try:
        python_says = ai_pause.is_paused()
    finally:
        project_paths.invalidate_local_settings_cache()

    code = (SCRIPTS_DIR / "run_ai_jobs.ps1").read_text(encoding="utf-8")
    start = code.index("function Get-AiPausedUntil")
    function = code[start:code.index("function Stop-RemoteGpuTunnel", start)]
    done = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", f"{function}\n'<' + (Get-AiPausedUntil -Path '{settings_file}') + '>'"],
        capture_output=True, text=True, timeout=60,
    )
    wrapper_says = done.stdout.strip() not in ("<>", "")
    assert "<" in done.stdout, done.stderr
    assert python_says is paused
    assert wrapper_says is paused
