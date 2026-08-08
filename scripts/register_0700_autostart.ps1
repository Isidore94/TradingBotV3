# Register (or refresh) the 07:00 weekday auto-launch task for TradingBotV3.
#
# Run this once per machine: it creates a Windows scheduled task that starts
# the GUI at 07:00 local every weekday in the logged-on user's session.
# StartWhenAvailable means a PC that boots at 07:40 still launches it.
# Together with the in-app 07:00 Auto Pilot self-arm this makes the whole
# chain hands-off: boot the machine -> GUI launches -> Auto Pilot arms ->
# BounceBot connects + scans -> scheduler runs the swing slots.
#
# The trigger also repeats every 15 minutes through the close (Tier A of
# docs/DURABILITY_CATCHUP_PLAN.md): a crash at 11:00 used to stay down until
# a human noticed, costing the whole afternoon's evidence. Each repetition
# re-runs launch_gui_auto.ps1, whose single-instance guard makes the relaunch
# idempotent -- when the GUI is already up the run exits immediately, so the
# only observable effect is that an outage self-heals within 15 minutes.
#
# Re-run this script once to pick the repetition up on an already-registered
# task (it unregisters and re-creates).
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

# Repeat every 15 minutes for 10 hours (07:00 -> 17:00 local), so any crash or
# missed boot self-heals for the whole session. Windows PowerShell 5.1 rejects
# -RepetitionInterval/-RepetitionDuration on a -Weekly trigger, so build a
# throwaway -Once trigger that accepts them and graft its .Repetition onto the
# weekly one before registration.
$repetition = (New-ScheduledTaskTrigger -Once -At 07:00 `
    -RepetitionInterval (New-TimeSpan -Minutes 15) `
    -RepetitionDuration (New-TimeSpan -Hours 10)).Repetition
$trigger.Repetition = $repetition

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings | Out-Null
Write-Output "Registered '$taskName': weekdays 07:00, repeating every 15 min for 10h (catch-up on late boot and mid-session crashes), launching $script"
