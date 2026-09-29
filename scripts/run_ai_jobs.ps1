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
# RTX 5080 box and this PC only tunnels to it. Prompts and answers travel over
# the ssh tunnel; every job still runs here and writes to the same AI store, so
# nothing about where results land changes. The endpoint URL points at the
# tunnel's local end (http://127.0.0.1:<port>/v1); the host side is always
# Ollama on its own 127.0.0.1:11434. Unset = today's behaviour, untouched.
$script:remoteAlias = ''
$script:tunnel = $null
$script:aiStoreDir = ''

function Invoke-RemoteGpuPreflight {
    param([string]$Alias, [int]$TunnelPort)
    $sshHost = ''
    foreach ($line in (& ssh -G $Alias)) { if ($line -like 'hostname *') { $sshHost = $line.Substring(9).Trim() } }
    if (-not $sshHost) { Write-Log "remote GPU: cannot resolve ssh alias '$Alias'; jobs will run degraded"; return }
    if (-not (Test-LocalEndpoint -EndpointHost $sshHost -Port 22)) {
        $wake = Join-Path $HOME 'bin\host-on.ps1'
        if (-not (Test-Path $wake)) { Write-Log "remote GPU: $sshHost is off and $wake is missing; jobs will run degraded"; return }
        Write-Log "remote GPU: $sshHost is off; sending Wake-on-LAN"
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $wake | Out-Null
        if (-not (Test-LocalEndpoint -EndpointHost $sshHost -Port 22)) { Write-Log "remote GPU: $sshHost did not wake; jobs will run degraded"; return }
    }
    if (-not (Test-LocalEndpoint -EndpointHost '127.0.0.1' -Port $TunnelPort)) {
        $script:tunnel = Start-Process -FilePath 'ssh' -WindowStyle Hidden -PassThru -ArgumentList @(
            '-N', '-L', "127.0.0.1:${TunnelPort}:127.0.0.1:11434",
            '-o', 'ExitOnForwardFailure=yes', '-o', 'BatchMode=yes',
            '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4', $Alias)
        $deadline = (Get-Date).AddSeconds(20)
        while ((Get-Date) -lt $deadline -and -not (Test-LocalEndpoint -EndpointHost '127.0.0.1' -Port $TunnelPort)) { Start-Sleep -Seconds 1 }
    }
    if (-not (Test-LocalEndpoint -EndpointHost '127.0.0.1' -Port $TunnelPort)) { Write-Log "remote GPU: tunnel to $Alias did not open on $TunnelPort; jobs will run degraded"; return }
    # The host-side start script is sent on stdin each run, so the host needs no
    # copy of it and cannot drift from the repo.
    $up = (Get-Content (Join-Path $PSScriptRoot 'remote_gpu\ollama_up.sh') -Raw) -replace "`r", ''
    $OutputEncoding = New-Object System.Text.UTF8Encoding($false)  # no BOM on the piped script
    $result = $up | & ssh -o BatchMode=yes -o ConnectTimeout=10 $Alias 'bash -s'
    Write-Log "remote GPU: $($result -join ' ') (tunnel 127.0.0.1:$TunnelPort -> $Alias)"
}

try {
    $settingsPath = Join-Path $env:LOCALAPPDATA 'TradingBotV3\local_settings.json'
    $endpoint = ''
    if (Test-Path $settingsPath) {
        $settings = Get-Content $settingsPath -Raw | ConvertFrom-Json
        $endpoint = $settings.ai_local_endpoint_url
        $script:remoteAlias = [string]$settings.ai_remote_gpu_ssh_alias
        $script:aiStoreDir = [string]$settings.ai_store_dir
    }
    if ([string]::IsNullOrWhiteSpace($endpoint)) {
        Write-Log "local inference: no endpoint configured; narration is off by design"
    } elseif (-not [string]::IsNullOrWhiteSpace($script:remoteAlias)) {
        Invoke-RemoteGpuPreflight -Alias $script:remoteAlias -TunnelPort ([System.Uri]$endpoint).Port
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

foreach ($stream in @(@{ Path = $stdout; Tag = 'out' }, @{ Path = $stderr; Tag = 'err' })) {
    if (Test-Path $stream.Path) {
        Get-Content $stream.Path | Where-Object { $_ -ne '' } | ForEach-Object {
            Add-Content -Path $logFile -Value "  [$($stream.Tag)] $_" -Encoding utf8
        }
        Remove-Item $stream.Path -Force -ErrorAction SilentlyContinue
    }
}

# Bring the host's model/GPU log back beside this run's log and into the AI
# store on the mini PC. Best effort: a mirror problem is never a job outcome.
if ($script:remoteAlias) {
    try {
        $hostLog = Join-Path $logDir ("gpu_host-" + (Get-Date -Format 'yyyyMMdd') + ".log")
        & ssh -o BatchMode=yes -o ConnectTimeout=10 $script:remoteAlias 'tail -n 500 ~/ollama-tradingbot.log; /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv' | Set-Content -Path $hostLog -Encoding utf8
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
if ($script:tunnel -and -not $script:tunnel.HasExited) { Stop-Process -Id $script:tunnel.Id -Force -ErrorAction SilentlyContinue }

$code = $process.ExitCode
switch ($code) {
    0       { Write-Log "=== AI jobs complete (exit 0: nothing due, or all jobs succeeded) ===" }
    1       { Write-Log "=== AI jobs FAILED (exit 1: at least one job failed) - see [err] lines above ===" }
    2       { Write-Log "=== AI jobs did not run (exit 2: AI store unreachable) ===" }
    default { Write-Log "=== AI jobs exited $code (0x$('{0:X}' -f $code)) ===" }
}

exit $code
