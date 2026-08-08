# Register (or refresh) the 06:00 weekday auto-launch task for TradingBotV3.
#
# Run this once per machine: it creates a Windows scheduled task that starts
# the GUI at 06:00 local every weekday in the logged-on user's session.
# StartWhenAvailable means a PC that boots at 06:40 still launches it.
# Together with the in-app Auto Pilot self-arm this makes the whole
# chain hands-off: boot the machine -> GUI launches -> Auto Pilot arms ->
# BounceBot connects + scans -> scheduler runs the swing slots.
#
# 06:00 LOCAL, not 07:00 (checkpoint review 2026-08-08, amendment 3.1): this
# desk is US Pacific, the NYSE open is 06:30 local, and the old 07:00 start
# meant the launcher -- and its self-heal repetition -- missed the first 30
# minutes of every session while idling four hours past the close. 06:00
# gives a 30-minute pre-open margin.
#
# The trigger also repeats every 15 minutes through the close (Tier A of
# docs/DURABILITY_CATCHUP_PLAN.md): a crash at 11:00 used to stay down until
# a human noticed, costing the whole afternoon's evidence. Each repetition
# re-runs launch_gui_auto.ps1, whose single-instance guard makes the relaunch
# idempotent -- when the GUI is already up the run exits immediately, so the
# only observable effect is that an outage self-heals within 15 minutes.
#
# Re-run this script once to pick the new time up on an already-registered
# task (it unregisters and re-creates).
#
# Remove with: Unregister-ScheduledTask -TaskName 'TradingBotV3 0700 Launch' -Confirm:$false

$taskName = "TradingBotV3 0700 Launch"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$script = Join-Path $root "scripts\launch_gui_auto.ps1"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`"" `
    -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At 06:00
# Anchor the start boundary a week in the past: with the boundary equal to the
# first occurrence, the scheduler treats it as strictly-after and silently
# skips registration day (observed 2026-07-10: Friday-morning registration
# computed Monday as the first run).
$trigger.StartBoundary = (Get-Date).AddDays(-7).Date.AddHours(6).ToString("yyyy-MM-dd'T'HH:mm:ss")

# Repeat every 15 minutes for 7.5 hours (06:00 -> 13:30 local = 30 min past the
# 13:00 close), so any crash or missed boot self-heals for the whole session
# without idling into the evening. Windows PowerShell 5.1 rejects
# -RepetitionInterval/-RepetitionDuration on a -Weekly trigger, so build a
# throwaway -Once trigger that accepts them and graft its .Repetition onto the
# weekly one before registration.
$repetition = (New-ScheduledTaskTrigger -Once -At 06:00 `
    -RepetitionInterval (New-TimeSpan -Minutes 15) `
    -RepetitionDuration (New-TimeSpan -Hours 7 -Minutes 30)).Repetition
$trigger.Repetition = $repetition

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings | Out-Null
Write-Output "Registered '$taskName': weekdays 06:00 local (Pacific; 30 min before the open), repeating every 15 min for 7.5h (self-heal through 30 min past the close), launching $script"
