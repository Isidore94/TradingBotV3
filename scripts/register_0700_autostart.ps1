# Register (or refresh) the 07:00 weekday auto-launch task for TradingBotV3.
#
# Run this once per machine (home PC, mini PC): it creates a Windows scheduled
# task that starts the GUI at 07:00 local every weekday in the logged-on
# user's session. StartWhenAvailable means a PC that boots at 07:40 still
# launches it. Together with the in-app 07:00 Auto Pilot self-arm this makes
# the whole chain hands-off: boot the machine -> GUI launches -> Auto Pilot
# arms -> BounceBot connects + scans -> scheduler runs the swing slots.
#
# Remove with: Unregister-ScheduledTask -TaskName 'TradingBotV3 0700 Launch' -Confirm:$false

$taskName = "TradingBotV3 0700 Launch"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$script = Join-Path $root "scripts\launch_gui_auto.ps1"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`"" `
    -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At 07:00
# Anchor the start boundary a week in the past: with the boundary equal to the
# first occurrence, the scheduler treats it as strictly-after and silently
# skips registration day (observed 2026-07-10: Friday-morning registration
# computed Monday as the first run).
$trigger.StartBoundary = (Get-Date).AddDays(-7).Date.AddHours(7).ToString("yyyy-MM-dd'T'HH:mm:ss")
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings | Out-Null
Write-Output "Registered '$taskName': weekdays 07:00 (catch-up on late boot), launching $script"
