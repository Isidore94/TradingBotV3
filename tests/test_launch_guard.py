"""Single-instance guard in scripts/launch_gui_auto.ps1.

The 06:00 task repeats every 15 minutes through the close (durability Tier A),
so this guard is asked "is a desk already up?" dozens of times a session, and
both wrong answers are expensive:

* a missed desk starts a second GUI -- double IB connection, duplicate bots;
* an invented desk means the launch never happens, and since the repetition
  asks the same question every 15 minutes, it keeps not happening. The 11:00
  crash that Tier A exists to self-heal would stay down all afternoon.

The guard matched Python processes only, so the frozen build
(dist/TradingBotV3/TradingBotV3.exe) looked like an idle machine and the task
would have started a second desk beside it (checkpoint review 2026-08-08
second review).

The PowerShell matcher carries its own fixture table behind ``-SelfTest``;
this test runs it, and separately pins the table itself so the cases cannot be
quietly deleted to make a failing matcher pass.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
GUARD = ROOT_DIR / "scripts" / "launch_gui_auto.ps1"

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows scheduled-task launcher"
)


def _run_selftest() -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(GUARD),
            "-SelfTest",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_guard_matcher_self_test_passes():
    result = _run_selftest()
    assert result.returncode == 0, (
        f"guard self-test failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "single-instance guard self-test cases passed" in result.stdout


def test_the_self_test_actually_covers_the_cases_that_matter():
    # A self-test is only worth running if it still asserts the things the
    # guard exists for, so pin the fixture table's content, not just its exit
    # code.
    source = GUARD.read_text(encoding="utf-8")

    # The frozen build: the defect this repair closes.
    assert "TradingBotV3.exe" in source
    assert "Name like 'TradingBotV3%'" in source
    # Both desk shapes the repo ships.
    assert "launch_gui.py" in source
    assert "scripts/gui.py --ui tk" in source
    # False positives are the expensive direction, so the negative cases stay.
    assert "not_launch_gui.py" in source
    assert "mygui.py" in source
    assert "-m pytest" in source


def test_detection_rests_on_a_running_process_not_a_leftover_file():
    # A stale heartbeat, or a writer lock that is unheld between publishes,
    # would make every 15-minute retry decline to launch. Detection is the
    # process table and nothing else; the only Test-Path calls happen *after*
    # the decision, checking the launcher's own paths.
    source = GUARD.read_text(encoding="utf-8")
    before_decision, _, after_decision = source.partition(
        "$already = Get-RunningTradingBotDesk"
    )
    assert "Get-CimInstance Win32_Process" in before_decision
    assert "Test-Path" not in before_decision
    assert "Test-Path" in after_decision
