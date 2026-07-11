# Auto-launch the TradingBotV3 GUI (used by the 07:00 scheduled task).
#
# Single-instance guard: a second GUI would double-connect to IB and run the
# bots twice, so if a python process is already running scripts\gui.py this
# exits quietly instead. The GUI itself then handles the rest of the 07:00
# hands-off chain (Auto Pilot self-arms, BounceBot connects, scanning starts,
# and project_paths waits for the Google Drive mount if it is racing boot).

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$already = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match 'gui\.py' }
if ($already) {
    Write-Output "TradingBotV3 GUI already running (pid $($already[0].ProcessId)) - nothing to do."
    exit 0
}

$launcher = Join-Path $root "TradingBotV3_GUI.cmd"
if (-not (Test-Path $launcher)) {
    Write-Output "Launcher not found: $launcher"
    exit 1
}
Write-Output "Launching TradingBotV3 GUI ($launcher)..."
Start-Process -FilePath $launcher -WorkingDirectory $root
exit 0
