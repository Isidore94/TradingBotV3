# Register (or refresh) the overnight AI batch task for TradingBotV3.
#
# Boots scripts/run_ai_jobs.py, which does its work and exits. It is separate
# from the trading GUI on purpose: the desk is meant to be up during market
# hours and this layer is meant to run when it is not, so hosting the batch
# jobs inside the GUI would couple two opposed lifecycles (and the GUI's own
# 07:00 task relaunches it every 15 minutes, which would orphan a long job).
#
# The trigger repeats every 30 minutes across the off-hours window rather than
# firing once. The runner is idempotent -- it asks the job ledger whether each
# job already completed for the session date -- so a repeat is a no-op on a
# healthy night and a self-heal on a night where the NAS was asleep or the
# endpoint was down at the first attempt. Same lesson as the desk's own
# 15-minute relaunch.
#
# IMPORTANT: the AI store is a UNC path (\\MINI-PC\...). A task running as
# SYSTEM has no network credentials and cannot reach it, so this registers to
# run as the logged-on user, matching the 07:00 desk task and Ollama's own
# startup shortcut.
#
# Remove with:
#   Unregister-ScheduledTask -TaskName 'TradingBotV3 AI Jobs' -Confirm:$false

param(
    # Defaults are the ET window (01:00-09:00) expressed in DESK LOCAL time,
    # because Task Scheduler triggers are local. On this Pacific desk that is
    # 22:00-06:00. Change both if the desk moves timezone.
    [string]$StartLocal = "22:00",
    [int]$DurationHours = 8,
    [int]$RepeatMinutes = 30
)

$taskName = "TradingBotV3 AI Jobs"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $python)) { $python = Join-Path $root ".venv\Scripts\python.exe" }
$script = Join-Path $root "scripts\run_ai_jobs.py"

if (-not (Test-Path $python)) { Write-Error "Virtual-environment Python not found: $python"; exit 1 }
if (-not (Test-Path $script)) { Write-Error "Runner not found: $script"; exit 1 }

$action = New-ScheduledTaskAction -Execute $python -Argument "`"$script`"" -WorkingDirectory $root

# Daily rather than weekly: overnight work spans midnight, and weekends are a
# legitimate time to run (the plan opens the window all day at weekends).
$trigger = New-ScheduledTaskTrigger -Daily -At $StartLocal
# Anchor the start boundary in the past so the scheduler does not treat the
# first occurrence as strictly-after and skip registration day -- the same
# footgun the 07:00 desk task hit on 2026-07-10.
$trigger.StartBoundary = (Get-Date).AddDays(-1).Date.Add([TimeSpan]::Parse($StartLocal)).ToString("yyyy-MM-dd'T'HH:mm:ss")

# Windows PowerShell 5.1 rejects -RepetitionInterval on a -Daily trigger, so
# build a throwaway -Once trigger that accepts it and graft its .Repetition on.
$repetition = (New-ScheduledTaskTrigger -Once -At $StartLocal `
    -RepetitionInterval (New-TimeSpan -Minutes $RepeatMinutes) `
    -RepetitionDuration (New-TimeSpan -Hours $DurationHours)).Repetition
$trigger.Repetition = $repetition

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew
# IgnoreNew is the skip-don't-pile-up policy in scheduler form: if a run is
# still going when the next repetition fires, the new one is dropped rather
# than stacked. Two runners would race the same ledger and the same endpoint.

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false }

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
    -RunLevel Limited | Out-Null

$endLocal = ([datetime]::ParseExact($StartLocal, "HH:mm", $null)).AddHours($DurationHours).ToString("HH:mm")
Write-Output "Registered '$taskName': daily $StartLocal-$endLocal local, repeating every $RepeatMinutes min."
Write-Output "Runner: $script"
Write-Output ""
Write-Output "Verify with:  .venv\Scripts\python.exe scripts\run_ai_jobs.py --status"
Write-Output "The runner refuses to launch during market hours regardless of this schedule."
