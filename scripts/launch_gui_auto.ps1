# Auto-launch the TradingBotV3 GUI (used by the 06:00 scheduled task).
#
# Single-instance guard: a second GUI would double-connect to IB and run the
# bots twice, so if the desk is already up this exits quietly instead. The GUI
# itself then handles the rest of the hands-off chain (Auto Pilot self-arms,
# BounceBot connects, scanning starts, and project_paths waits for the Google
# Drive mount if it is racing boot).
#
# The trigger repeats every 15 minutes through the close, so this guard runs
# dozens of times a session and both of its failure directions cost real money:
#
#   * missing a running desk  -> a second GUI, double IB connect, duplicate bots;
#   * inventing a running desk -> the launch never happens, and because the
#     repetition asks the same question every 15 minutes it keeps not happening.
#     A crash at 11:00 then stays down for the rest of the session, which is the
#     exact outage Tier A exists to end.
#
# So detection is based only on a process that is actually running right now,
# never on a file left behind by one.
#
# Why not ask the app itself (checkpoint review 2026-08-08 second review):
# neither piece of the app's own machinery answers "is a desk up?".
#   * scripts/local_writer_lock.py is a *per-publish* lock. Its named mutex and
#     lock file are acquired around a single verified publish and released
#     immediately after, so it is unheld almost all of the time a desk is
#     running - it would report "nothing running" nearly every time it is asked.
#   * diagnostics/heartbeat.json carries a pid, but only Auto Pilot writes it, so
#     a desk with Auto Pilot off never appears; and a stale file from a killed
#     process would have to be pid-checked against a live process anyway, which
#     is what the check below already does directly.
# A process-name union is therefore the authoritative available signal. It
# covers both shapes the desk ships in: the repo checkout run under the venv
# Python, and the frozen onedir build (dist/TradingBotV3/TradingBotV3.exe),
# which the old guard missed entirely - it matched python processes only, so
# the exe running the desk looked like an idle machine and the task would have
# started a second one beside it.
#
# -SelfTest runs the matcher over synthetic process records and exits non-zero
# on a mismatch; tests/test_launch_guard.py drives it.

[CmdletBinding()]
param(
    [switch]$SelfTest
)

# Frozen-build process names. onedir keeps the exe name, so this is the whole
# list; a renamed copy is out of scope (and would also break the task's own
# path).
$script:FrozenProcessNames = @('TradingBotV3', 'TradingBotV3.exe')

function Test-TradingBotDeskProcess {
    <#
      .SYNOPSIS
      True when a process record is a running TradingBotV3 desk.

      .DESCRIPTION
      Two shapes count:
        * a Python process whose command line runs launch_gui.py (the PySide6
          desk) or scripts/gui.py (the legacy Tk UI kept during migration);
        * the frozen build's own executable, whatever its command line.
      Nothing else does - in particular a bare `python` with no launcher on its
      command line is somebody's REPL, not the desk.
    #>
    param(
        [string]$Name,
        [string]$CommandLine
    )

    $processName = ($Name -as [string])
    if (-not $processName) { return $false }

    $bare = [System.IO.Path]::GetFileNameWithoutExtension($processName)
    if ($script:FrozenProcessNames -contains $processName -or
        $script:FrozenProcessNames -contains $bare) {
        return $true
    }

    if ($bare -like 'python*') {
        # Anchored on a path separator or a string boundary so `mygui.py` and
        # `not_launch_gui.py` do not count.
        return ($CommandLine -match '(^|[\s"''\\/])(launch_gui|gui)\.py(\s|"|''|$)')
    }

    return $false
}

function Get-RunningTradingBotDesk {
    <#
      .SYNOPSIS
      The first running desk process, or $null.
    #>
    $candidates = @()
    foreach ($filter in @("Name like 'python%'", "Name like 'TradingBotV3%'")) {
        try {
            $candidates += @(Get-CimInstance Win32_Process -Filter $filter -ErrorAction Stop)
        } catch {
            Write-Output "Could not enumerate processes ($filter): $($_.Exception.Message)"
        }
    }
    foreach ($process in $candidates) {
        if (Test-TradingBotDeskProcess -Name $process.Name -CommandLine $process.CommandLine) {
            return $process
        }
    }
    return $null
}

function Invoke-GuardSelfTest {
    $cases = @(
        @{ Name = 'python.exe';       Cmd = 'C:\p\.venv\Scripts\python.exe C:\p\launch_gui.py'; Expect = $true;  Why = 'venv python running the PySide6 launcher' },
        @{ Name = 'pythonw.exe';      Cmd = '"C:\p\.venv\Scripts\pythonw.exe" "C:\p\launch_gui.py"'; Expect = $true; Why = 'quoted pythonw launch' },
        @{ Name = 'python.exe';       Cmd = 'python scripts/gui.py --ui tk'; Expect = $true;  Why = 'legacy Tk UI still counts as a desk' },
        @{ Name = 'TradingBotV3.exe'; Cmd = 'C:\p\dist\TradingBotV3\TradingBotV3.exe'; Expect = $true; Why = 'frozen build - the case the old guard missed' },
        @{ Name = 'TradingBotV3';     Cmd = ''; Expect = $true;  Why = 'frozen build reported without its extension' },
        @{ Name = 'python.exe';       Cmd = 'C:\p\.venv\Scripts\python.exe -m pytest tests/ -q'; Expect = $false; Why = 'the test suite is not a desk' },
        @{ Name = 'python.exe';       Cmd = 'python C:\other\not_launch_gui.py'; Expect = $false; Why = 'substring match must not count' },
        @{ Name = 'python.exe';       Cmd = 'python C:\other\mygui.py'; Expect = $false; Why = 'substring match must not count' },
        @{ Name = 'python.exe';       Cmd = ''; Expect = $false; Why = 'a bare REPL is not a desk' },
        @{ Name = 'notepad.exe';      Cmd = 'notepad launch_gui.py'; Expect = $false; Why = 'editing the launcher is not running it' },
        @{ Name = '';                 Cmd = 'launch_gui.py'; Expect = $false; Why = 'nameless record' }
    )
    # Messages go to the output stream, so the exit code travels in a script
    # variable rather than as a return value the messages would be mixed into.
    $failures = 0
    foreach ($case in $cases) {
        $actual = Test-TradingBotDeskProcess -Name $case.Name -CommandLine $case.Cmd
        if ($actual -ne $case.Expect) {
            $failures++
            Write-Output "FAIL expected=$($case.Expect) actual=$actual  [$($case.Why)]  name=$($case.Name) cmd=$($case.Cmd)"
        }
    }
    if ($failures -gt 0) {
        Write-Output "$failures of $($cases.Count) single-instance guard self-test case(s) failed."
        $script:SelfTestExitCode = 1
        return
    }
    Write-Output "$($cases.Count)/$($cases.Count) single-instance guard self-test cases passed."
    $script:SelfTestExitCode = 0
}

if ($SelfTest) {
    $script:SelfTestExitCode = 1
    Invoke-GuardSelfTest
    exit $script:SelfTestExitCode
}

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$already = Get-RunningTradingBotDesk
if ($already) {
    Write-Output "TradingBotV3 GUI already running ($($already.Name), pid $($already.ProcessId)) - nothing to do."
    exit 0
}

$python = Join-Path $root ".venv\Scripts\python.exe"
$launcher = Join-Path $root "launch_gui.py"
if (-not (Test-Path $python)) {
    Write-Output "Virtual-environment Python not found: $python"
    exit 1
}
if (-not (Test-Path $launcher)) {
    Write-Output "Launcher not found: $launcher"
    exit 1
}
Write-Output "Launching TradingBotV3 GUI ($launcher)..."
Start-Process -FilePath $python -ArgumentList @($launcher) -WorkingDirectory $root
exit 0
