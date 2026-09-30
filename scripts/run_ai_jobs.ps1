# Wrapper for the "TradingBotV3 AI Jobs" scheduled task.
#
# WHY THIS EXISTS
# ---------------
# The task originally ran `pythonw.exe scripts\run_ai_jobs.py` directly and
# exited 0xC0000142 (STATUS_DLL_INIT_FAILED) on 2026-08-10, so the whole
# overnight AI layer silently did nothing. Two separate problems produced that:
#
#   1. `pythonw.exe` is a GUI-subsystem binary. It needs a window station to
#      initialize, which a task starting at 06:00 on a locked/waking session may
#      not have. `run_ai_jobs.py` imports no Qt and never opens a window, so the
#      GUI subsystem bought nothing and cost the whole run.
#   2. `pythonw.exe` discards stdout and stderr. The failure therefore left no
#      message anywhere - only a hex code in Task Scheduler's history, hours
#      after the fact. A batch layer that fails silently is indistinguishable
#      from one that had nothing to do.
#
# So: console `python.exe`, output captured to a dated log, and the child's real
# exit code propagated to the scheduler. Hidden window styling belongs on the
# task action (-WindowStyle Hidden), not here, matching launch_gui_auto.ps1.
#
# EXIT CODES are run_ai_jobs.py's own and must survive unchanged:
#   0 = nothing due, or every job succeeded
#   1 = at least one job failed
#   2 = the AI store was unreachable, so nothing ran
# Anything this wrapper itself refuses on exits 3, so a wrapper problem is never
# mistaken for a job result.

[CmdletBinding()]
param(
    # Passed through to run_ai_jobs.py (e.g. --status, --slot ai_summary, --force).
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Passthrough
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = Join-Path $root '.venv\Scripts\python.exe'
$script = Join-Path $root 'scripts\run_ai_jobs.py'

$logDir = Join-Path $env:LOCALAPPDATA 'TradingBotV3\logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logFile = Join-Path $logDir ("ai_jobs-" + (Get-Date -Format 'yyyyMMdd') + ".log")

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Message"
    Write-Output $line
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

# A missing interpreter or entry point is a wrapper-level refusal, not a job
# outcome: say which path was missing, because "it didn't run" with no path is
# the failure mode this file was written to end.
if (-not (Test-Path $python))  { Write-Log "REFUSED: venv Python not found at $python"; exit 3 }
if (-not (Test-Path $script))  { Write-Log "REFUSED: entry point not found at $script"; exit 3 }

# The scheduled task passes nothing, so the no-argument case IS the routine
# nightly run - not a caller that forgot something. "(no arguments)" read like a
# defect in the one log line an operator sees most often, which is the opposite
# of what this wrapper exists to do.
$argLine = if ($Passthrough) { $Passthrough -join ' ' } else { 'scheduled run: every due slot' }
Write-Log "=== AI jobs starting === $argLine"

# ---------------------------------------------------------------------------
# Local inference preflight (2026-08-28)
# ---------------------------------------------------------------------------
# The local model server is the narration half of this layer. It is a
# user-session tray app with NO autostart entry, so a desk restart silently
# ends it: on 2026-08-27 its log stopped at 06:12, the desk restarted around
# 13:00, and all three narrating jobs spent the whole 22:00-06:00 window
# retrying against a refused connection. The deterministic jobs were fine, which
# is exactly why nobody noticed until the summaries were read.
#
# An unattended nightly must not depend on a human having clicked something. So:
# probe the port, start the server if it is down, and CARRY ON either way. This
# never refuses the run - `degraded_no_narrative` is a designed state, the fact
# packs and the counting jobs do not need a model, and a preflight that could
# block the night would be worse than the problem it fixes.
function Test-LocalEndpoint {
    param([string]$EndpointHost, [int]$Port)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $wait = $client.BeginConnect($EndpointHost, $Port, $null, $null)
        if (-not $wait.AsyncWaitHandle.WaitOne(1500, $false)) { return $false }
        $client.EndConnect($wait)
        return $true
    } catch { return $false } finally { $client.Close() }
}

# Remote GPU host (2026-09-28)
# ----------------------------
# With `ai_remote_gpu_ssh_alias` set (e.g. "claude-host"), the model runs on the
# RTX 5080 box and this PC only tunnels to it. Every job still runs here and
# writes to the same AI store. The saved endpoint setting is NOT changed: once
# the tunnel is up, the model tag is present and a warm-up call answered, the
# child gets TRADINGBOTV3_AI_ENDPOINT_OVERRIDE pointing at the tunnel. Any
# failure leaves the override unset and falls through to the local server.
#
# Night flags (one night = the evening's date, so 22:00-05:30 share one key):
#   gpu_host_dead-<night>.flag   the host failed tonight (preflight, or the job
#                                lost it mid-run); later firings go local at once.
#   gpu_host_woken-<night>.flag  this job woke the host, so this job powers it off
#                                once the night is done (backstop: after
#                                `ai_remote_gpu_off_after`, 05:30).
#   night_passes-<night>.txt     scheduled passes that ran tonight.
#   night_done-<night>.flag      a clean pass, or the pass plus ONE recheck, ran.
#                                Later firings exit at once and wake nothing.
# A host that was already up is left on (someone else may be using it); only
# its model is unloaded.
$script:remoteAlias = ''
$script:watchdog = $null
$script:aiStoreDir = ''
$nightKey = (Get-Date).AddHours(-12).ToString('yyyyMMdd')
$stateDir = Join-Path $env:LOCALAPPDATA 'TradingBotV3'
$script:deadFlag = Join-Path $stateDir "gpu_host_dead-$nightKey.flag"
$script:wokenFlag = Join-Path $stateDir "gpu_host_woken-$nightKey.flag"
$script:passFile = Join-Path $stateDir "night_passes-$nightKey.txt"
$script:doneFlag = Join-Path $stateDir "night_done-$nightKey.flag"
$scheduled = -not ($Passthrough | Where-Object { $_ })
# These runs never call a model, so they never wake or load the 5080.
$noModelRun = [bool]($Passthrough | Where-Object { $_ -in @('--retry-journal-import', '--status') })

# Runs one command on the host with a hard timeout and stdin closed. Ok is
# $false when ssh could not start or outlived the timeout.
function Invoke-HostSsh {
    param([string]$Command, [string]$InputText = '', [int]$TimeoutSeconds = 60)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = 'ssh.exe'
    $psi.Arguments = "-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 $script:remoteAlias `"$($Command -replace '"', '\"')`""
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    try { $proc = [System.Diagnostics.Process]::Start($psi) } catch {
        Write-Log "remote GPU: ssh did not start ($($_.Exception.Message))" | Out-Null
        return [pscustomobject]@{ Ok = $false; Lines = @() }
    }
    $out = $proc.StandardOutput.ReadToEndAsync()
    $null = $proc.StandardError.ReadToEndAsync()
    if ($InputText) { $proc.StandardInput.Write($InputText) }
    $proc.StandardInput.Close()
    if (-not $proc.WaitForExit($TimeoutSeconds * 1000)) {
        try { $proc.Kill() } catch { $null = $_ }
        Write-Log "remote GPU: ssh timed out after ${TimeoutSeconds}s: $Command" | Out-Null
        return [pscustomobject]@{ Ok = $false; Lines = @() }
    }
    $lines = @(([string]$out.Result) -split "`r?`n" | Where-Object { $_ -ne '' })
    return [pscustomobject]@{ Ok = $true; Lines = $lines }
}

function Invoke-RemoteGpuPreflight {
    param([string]$Alias, [int]$TunnelPort, [string]$Model)
    $sshHost = ''
    foreach ($line in (& ssh -G $Alias)) { if ($line -like 'hostname *') { $sshHost = $line.Substring(9).Trim() } }
    if (-not $sshHost) { Write-Log "remote GPU: cannot resolve ssh alias '$Alias'; using the local server"; return $false }
    if (-not (Test-LocalEndpoint -EndpointHost $sshHost -Port 22)) {
        $wake = Join-Path $HOME 'bin\host-on.ps1'
        if (-not (Test-Path $wake)) { Write-Log "remote GPU: $sshHost is off and $wake is missing; using the local server"; return $false }
        Write-Log "remote GPU: $sshHost is off; sending Wake-on-LAN"
        # host-on.ps1 itself polls port 22 for up to 5 minutes.
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $wake | Out-Null
        if (-not (Test-LocalEndpoint -EndpointHost $sshHost -Port 22)) { Write-Log "remote GPU: $sshHost did not wake within 5 minutes; using the local server"; return $false }
        Set-Content -Path $script:wokenFlag -Value (Get-Date -Format s) -Encoding ascii
    } else {
        Write-Log "remote GPU: $sshHost is already up"
    }
    # The host-side start script is sent on stdin each run, so the host needs no
    # copy of it and cannot drift from the repo.
    $up = (Get-Content (Join-Path $PSScriptRoot 'remote_gpu\ollama_up.sh') -Raw) -replace "`r", ''
    # PowerShell 5.1 may prefix a BOM when piping to a native exe; strip it host-side.
    $result = Invoke-HostSsh -Command "sed '1s/^\xEF\xBB\xBF//' | bash -s" -InputText $up -TimeoutSeconds 90
    if (-not $result.Ok) { Write-Log "remote GPU: the host start script did not answer; using the local server"; return $false }
    Write-Log "remote GPU: $($result.Lines -join ' ')"
    # Watchdog: a hidden loop that reopens the tunnel whenever ssh exits, so a
    # network blip costs seconds instead of the rest of the night.
    if (-not (Test-LocalEndpoint -EndpointHost '127.0.0.1' -Port $TunnelPort)) {
        $loop = "while (`$true) { ssh -N -L 127.0.0.1:${TunnelPort}:127.0.0.1:11434 -o ExitOnForwardFailure=yes -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 $Alias; Start-Sleep -Seconds 2 }"
        $script:watchdog = Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden -PassThru -ArgumentList @('-NoProfile', '-Command', $loop)
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline -and -not (Test-LocalEndpoint -EndpointHost '127.0.0.1' -Port $TunnelPort)) { Start-Sleep -Seconds 1 }
    }
    if (-not (Test-LocalEndpoint -EndpointHost '127.0.0.1' -Port $TunnelPort)) { Write-Log "remote GPU: tunnel to $Alias did not open on $TunnelPort; using the local server"; return $false }
    $base = "http://127.0.0.1:$TunnelPort"
    try {
        $tags = (Invoke-RestMethod "$base/api/tags" -TimeoutSec 15).models | ForEach-Object { $_.name }
        if ($tags -notcontains $Model -and $tags -notcontains "${Model}:latest") { Write-Log "remote GPU: model '$Model' is not on $Alias; using the local server"; return $false }
        # Load the model now, so the jobs' own short probe never pays a cold load.
        $warm = @{ model = $Model; prompt = 'ok'; stream = $false; keep_alive = '24h'; options = @{ num_predict = 1 } } | ConvertTo-Json
        $sw = [Diagnostics.Stopwatch]::StartNew()
        Invoke-RestMethod "$base/api/generate" -Method Post -Body $warm -ContentType 'application/json' -TimeoutSec 300 | Out-Null
        Write-Log ("remote GPU: '$Model' warm in {0:n0}s" -f $sw.Elapsed.TotalSeconds)
    } catch { Write-Log "remote GPU: warm-up failed ($($_.Exception.Message)); using the local server"; return $false }
    return $true
}

function Stop-RemoteGpuTunnel {
    if (-not $script:watchdog) { return }
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$($script:watchdog.Id)" -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Stop-Process -Id $script:watchdog.Id -Force -ErrorAction SilentlyContinue
}

if ($scheduled -and (Test-Path $script:doneFlag)) {
    Write-Log "night done: tonight's pass and recheck already ran; nothing to do"
    Write-Log "=== AI jobs complete (exit 0: night already done) ==="
    exit 0
}

try {
    $settingsPath = Join-Path $env:LOCALAPPDATA 'TradingBotV3\local_settings.json'
    $endpoint = ''
    $remoteReady = $false
    $aiPausedUntil = ''
    if (Test-Path $settingsPath) {
        $settings = Get-Content $settingsPath -Raw | ConvertFrom-Json
        $endpoint = $settings.ai_local_endpoint_url
        $script:remoteAlias = [string]$settings.ai_remote_gpu_ssh_alias
        $script:aiStoreDir = [string]$settings.ai_store_dir
        # Pause AI: a future `ai_paused_until` means this run leaves the 5080 alone.
        try {
            if ($settings.ai_paused_until -and [DateTimeOffset]::Parse([string]$settings.ai_paused_until) -gt [DateTimeOffset]::Now) {
                $aiPausedUntil = [string]$settings.ai_paused_until
            }
        } catch { $null = $_ }
    }
    if ($aiPausedUntil) {
        # No WOL, no host script, no tunnel, no warm-up, no override, no mirror and no power-off.
        $noModelRun = $true
        Write-Log "AI paused until $aiPausedUntil; remote GPU untouched"
    } elseif ([string]::IsNullOrWhiteSpace($endpoint)) {
        Write-Log "local inference: no endpoint configured; narration is off by design"
    } elseif ($noModelRun -and -not [string]::IsNullOrWhiteSpace($script:remoteAlias)) {
        Write-Log "remote GPU: this run needs no model; the host is not touched"
    } elseif (-not [string]::IsNullOrWhiteSpace($script:remoteAlias) -and (Test-Path $script:deadFlag)) {
        Write-Log "remote GPU: the host failed earlier tonight; using the local server"
    } elseif (-not [string]::IsNullOrWhiteSpace($script:remoteAlias)) {
        $tunnelPort = if ($settings.ai_remote_gpu_tunnel_port) { [int]$settings.ai_remote_gpu_tunnel_port } else { 11435 }
        $model = if ($settings.ai_local_model_medium) { [string]$settings.ai_local_model_medium } else { 'gemma3:12b' }
        # Write-Log also writes to the pipeline, so only the LAST value is the verdict.
        $preflight = @(Invoke-RemoteGpuPreflight -Alias $script:remoteAlias -TunnelPort $tunnelPort -Model $model)
        $remoteReady = ($preflight.Count -gt 0) -and ($preflight[-1] -is [bool]) -and $preflight[-1]
        if ($remoteReady) {
            $env:TRADINGBOTV3_AI_ENDPOINT_OVERRIDE = "http://127.0.0.1:$tunnelPort/v1"
            # The job writes this flag if it loses the host mid-run and goes local.
            $env:TRADINGBOTV3_AI_REMOTE_DEAD_FLAG = $script:deadFlag
            Write-Log "remote GPU: this run uses the 5080 ($env:TRADINGBOTV3_AI_ENDPOINT_OVERRIDE)"
        } else {
            Stop-RemoteGpuTunnel
            Set-Content -Path $script:deadFlag -Value (Get-Date -Format s) -Encoding ascii
        }
    }
    # The local server is always made ready: it is the fallback when the host
    # drops mid-run, and it loads no model until a request reaches it.
    if ([string]::IsNullOrWhiteSpace($endpoint)) {
        # Nothing to start locally.
    } else {
        $uri = [System.Uri]$endpoint
        # Only a LOCAL server is ours to start. A remote endpoint belongs to
        # whoever runs it, and reaching for a process here would be wrong.
        if ($uri.Host -notin @('127.0.0.1', 'localhost', '::1')) {
            Write-Log "local inference: endpoint $($uri.Host) is remote; not starting anything"
        } elseif (Test-LocalEndpoint -EndpointHost $uri.Host -Port $uri.Port) {
            Write-Log "local inference: server already listening on $($uri.Host):$($uri.Port)"
        } else {
            $ollama = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
            if (-not (Test-Path $ollama)) {
                $found = Get-Command ollama -ErrorAction SilentlyContinue
                if ($found) { $ollama = $found.Source }
            }
            if (-not (Test-Path $ollama)) {
                Write-Log "local inference: DOWN on $($uri.Host):$($uri.Port) and ollama.exe was not found; jobs will run degraded"
            } else {
                Write-Log "local inference: DOWN on $($uri.Host):$($uri.Port); starting $ollama serve"
                Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden | Out-Null
                # Model load happens on first request, not at listen, so this
                # waits only for the socket. 60s is generous for that.
                $deadline = (Get-Date).AddSeconds(60)
                $up = $false
                while ((Get-Date) -lt $deadline) {
                    if (Test-LocalEndpoint -EndpointHost $uri.Host -Port $uri.Port) { $up = $true; break }
                    Start-Sleep -Seconds 2
                }
                if ($up) {
                    Write-Log "local inference: server came up; narration is available this run"
                } else {
                    Write-Log "local inference: server did NOT come up within 60s; jobs will run degraded"
                }
            }
        }
    }
} catch {
    # A preflight fault is never a job outcome. Say what happened and continue.
    Write-Log "local inference: preflight error (continuing anyway): $($_.Exception.Message)"
}

# Redirect both streams into the log. `2>&1` on a native exe is avoided inside
# PowerShell 5.1 (it wraps stderr lines in ErrorRecords and falsifies $?), so
# the redirection is done by Start-Process at the OS level instead, and the two
# streams are appended to the shared log afterwards.
$stdout = Join-Path $logDir 'ai_jobs.stdout.tmp'
$stderr = Join-Path $logDir 'ai_jobs.stderr.tmp'

# Built by filtering rather than concatenating: `@($script) + $null` yields an
# array WITH a null element, and Start-Process -ArgumentList rejects that. The
# scheduled task passes no arguments at all, so that is the normal path, not an
# edge case - it is how the first wrapper build failed its own task run.
$arguments = @($script) + @($Passthrough | Where-Object { $_ })
$process = Start-Process -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $root `
    -NoNewWindow `
    -Wait `
    -PassThru `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr

$summaryLine = ''
foreach ($stream in @(@{ Path = $stdout; Tag = 'out' }, @{ Path = $stderr; Tag = 'err' })) {
    if (Test-Path $stream.Path) {
        Get-Content $stream.Path | Where-Object { $_ -ne '' } | ForEach-Object {
            if ($_ -match 'AI jobs for session .*: \d+ ok, ') { $summaryLine = $_ }
            Add-Content -Path $logFile -Value "  [$($stream.Tag)] $_" -Encoding utf8
        }
        Remove-Item $stream.Path -Force -ErrorAction SilentlyContinue
    }
}
$code = $process.ExitCode

# Bring the host's model/GPU log back beside this run's log and into the AI
# store on the mini PC. Best effort: a mirror problem is never a job outcome.
if ($script:remoteAlias -and -not $noModelRun) {
    try {
        $hostLog = Join-Path $logDir ("gpu_host-" + (Get-Date -Format 'yyyyMMdd') + ".log")
        $hostRead = Invoke-HostSsh -Command 'tail -n 500 ~/ollama-tradingbot.log; /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv' -TimeoutSeconds 60
        # An unreachable host must not overwrite the last good copy with nothing.
        if (-not $hostRead.Ok -or $hostRead.Lines.Count -eq 0) { throw "host unreachable, kept the last copy" }
        $hostRead.Lines | Set-Content -Path $hostLog -Encoding utf8
        if ($script:aiStoreDir -and (Test-Path $script:aiStoreDir)) {
            $mirror = Join-Path $script:aiStoreDir 'gpu_host'
            New-Item -ItemType Directory -Path $mirror -Force | Out-Null
            Copy-Item -Path $hostLog -Destination $mirror -Force
            Write-Log "remote GPU: host log mirrored to $mirror"
        } else {
            Write-Log "remote GPU: AI store not reachable; host log kept at $hostLog only"
        }
    } catch { Write-Log "remote GPU: host log mirror failed (ignored): $($_.Exception.Message)" }
}

# One pass, one recheck (trader 2026-09-29). A pass that ran (exit 0 or 1)
# counts; a clean pass ends the night, otherwise the next firing is the one
# recheck and the night ends after it. Exit 2 (store unreachable) ran nothing.
$nightDone = $false
if ($scheduled -and $code -in @(0, 1)) {
    try {
        $passes = 1
        if (Test-Path $script:passFile) { $passes += [int](([string](Get-Content $script:passFile -Raw)).Trim()) }
        Set-Content -Path $script:passFile -Value $passes -Encoding ascii
        $clean = ($code -eq 0) -and ($summaryLine -match ', 0 degraded, 0 failed, ')
        if ($clean -or $passes -ge 2) {
            $nightDone = $true
            Set-Content -Path $script:doneFlag -Value (Get-Date -Format s) -Encoding ascii
            $why = if ($clean) { 'every job came out clean' } else { 'the one recheck ran' }
            Write-Log "night done after pass ${passes}: $why; later firings tonight do nothing"
        } else {
            Write-Log "night pass $passes had failed or degraded jobs; the next firing is the one recheck"
        }
    } catch { Write-Log "night pass count failed (ignored): $($_.Exception.Message)" }
}
# Backstop: a night still going at `ai_remote_gpu_off_after` (05:30) is over.
$offAfter = if ($settings.ai_remote_gpu_off_after) { [string]$settings.ai_remote_gpu_off_after } else { '05:30' }
$now = Get-Date
$morning = $now.TimeOfDay -ge ([datetime]::ParseExact($offAfter, 'HH:mm', $null)).TimeOfDay -and $now.Hour -lt 12
$hostFinished = $script:remoteAlias -and -not $noModelRun -and ($nightDone -or ($scheduled -and $morning))

# Free the 5080's memory as soon as the night is over, even on a host left on.
if ($hostFinished -and $remoteReady) {
    try {
        $unload = @{ model = $model; keep_alive = 0 } | ConvertTo-Json
        Invoke-RestMethod "http://127.0.0.1:$tunnelPort/api/generate" -Method Post -Body $unload -ContentType 'application/json' -TimeoutSec 30 | Out-Null
        Write-Log "remote GPU: '$model' unloaded from the host"
    } catch { Write-Log "remote GPU: model unload failed (ignored): $($_.Exception.Message)" }
}
Stop-RemoteGpuTunnel

# Power the host off once the night is over, only if this job woke it and
# nothing else is running there.
if ($hostFinished -and (Test-Path $script:wokenFlag)) {
    try {
        $busy = Invoke-HostSsh -Command "tmux ls -F '#S' 2>/dev/null | grep -v '^ollama$'" -TimeoutSeconds 30
        if (-not $busy.Ok) {
            Write-Log "remote GPU: host left ON - could not check it for other work"
        } elseif ($busy.Lines.Count -gt 0) {
            Write-Log "remote GPU: host left ON - other work is running there: $($busy.Lines -join ', ')"
        } else {
            $off = Invoke-HostSsh -Command "/mnt/c/Windows/System32/shutdown.exe /s /t 60 /c 'TradingBotV3 night AI done'" -TimeoutSeconds 30
            if ($off.Ok) { Write-Log "remote GPU: night done, logs copied; host shutdown requested (60s)" }
            else { Write-Log "remote GPU: host shutdown request did not answer" }
        }
        Remove-Item $script:wokenFlag -Force -ErrorAction SilentlyContinue
    } catch { Write-Log "remote GPU: shutdown step failed (ignored): $($_.Exception.Message)" }
}

switch ($code) {
    0       { Write-Log "=== AI jobs complete (exit 0: nothing due, or all jobs succeeded) ===" }
    1       { Write-Log "=== AI jobs FAILED (exit 1: at least one job failed) - see [err] lines above ===" }
    2       { Write-Log "=== AI jobs did not run (exit 2: AI store unreachable) ===" }
    default { Write-Log "=== AI jobs exited $code (0x$('{0:X}' -f $code)) ===" }
}

exit $code
