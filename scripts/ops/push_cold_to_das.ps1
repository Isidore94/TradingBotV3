<#
    push_cold_to_das.ps1

    Pushes the COLD, append-only, unboundedly-growing parts of the TradingBotV3
    shared store to the MINI-PC DAS. Hot state (the setup tracker, the big
    runtime CSVs) is deliberately NOT pushed: it is rewritten constantly, some of
    it as ~750 MB atomic replaces, and it must stay on local SSD.

    TWO JOBS, TWO SCOPES. DO NOT MERGE THEM.
      THIS is the hourly, incremental, undated mirror of cold subtrees.
      snapshot_to_das.ps1 (R10.A) is the nightly DATED SNAPSHOT of the hot state
      this one excludes on purpose - data\runtime, the home-root evidence files,
      _tools and the machine-local diagnostics tree. It copies; it never moves.
      Neither is a substitute for the other: this one has no dates to roll back
      to, and that one is not continuous.

    If MINI-PC is unreachable or not writable, this exits cleanly (code 0) and
    everything simply stays staged on the local disk until the next run. That is
    the intended fallback, not an error.

    Run it from Task Scheduler, e.g. hourly:
      powershell -NoProfile -ExecutionPolicy Bypass -File C:\TradingBotData\_tools\push_cold_to_das.ps1
#>

[CmdletBinding()]
param(
    [string] $Source = 'C:\TradingBotData',
    [string] $Dest   = '\\MINI-PC\Trading Bot Data',
    [switch] $WhatIfOnly
)

$ErrorActionPreference = 'Stop'
$logDir = Join-Path $Source '_tools\logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$stamp   = Get-Date -Format 'yyyyMMdd-HHmmss'
$logFile = Join-Path $logDir "push-$stamp.log"

function Write-Log {
    param([string] $Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Message"
    Write-Output $line
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

# Cold subtrees: relative path under $Source. These only ever grow and are read
# far more often than written, so SMB latency is acceptable.
$coldSubtrees = @(
    'data\daily_bars',
    'data\intraday_bars',
    'output',
    'logs',
    'away_report_archive',
    # R10 month-segmented evidence ledgers. Append-only and unboundedly growing,
    # which is exactly this job's shape - unlike everything else in data\runtime,
    # which is hot and belongs to the nightly snapshot instead. Absent subtrees
    # are skipped below, so these are safe to list before the ledgers exist.
    'data\runtime\evidence_ledgers'
)

Write-Log "=== cold push starting ==="
Write-Log "source: $Source"
Write-Log "dest  : $Dest"

# ---- preflight: is the DAS actually usable? ------------------------------
$staged = 0L
foreach ($rel in $coldSubtrees) {
    $p = Join-Path $Source $rel
    if (Test-Path $p) {
        $staged += (Get-ChildItem $p -Recurse -File -Force -ErrorAction SilentlyContinue |
                    Measure-Object -Property Length -Sum).Sum
    }
}
$stagedMB = [math]::Round($staged / 1MB, 1)

if (-not (Test-Path $Dest)) {
    Write-Log "MINI-PC UNREACHABLE - $stagedMB MB stays staged locally. Not an error; will retry next run."
    exit 0
}

$probe = Join-Path $Dest ".write_probe_$stamp"
try {
    Set-Content -Path $probe -Value 'probe' -Encoding utf8 -ErrorAction Stop
    Remove-Item $probe -Force -ErrorAction SilentlyContinue
} catch {
    Write-Log "MINI-PC REACHABLE BUT NOT WRITABLE ($($_.Exception.Message.Trim()))."
    Write-Log "$stagedMB MB stays staged locally. Fix share/NTFS permissions on MINI-PC, then re-run."
    exit 0
}

Write-Log "DAS writable. $stagedMB MB of cold data to reconcile."
if ($WhatIfOnly) { Write-Log 'WhatIfOnly set - stopping before copy.'; exit 0 }

# ---- copy ----------------------------------------------------------------
# /E   include empty dirs        /Z  restartable (survives a network blip)
# /XO  never overwrite a newer file already on the DAS
# No /MIR and no /MOV: nothing on the DAS is ever deleted, and the local copy
# stays put so the running bot keeps reading it.
$failed = @()
foreach ($rel in $coldSubtrees) {
    $src = Join-Path $Source $rel
    if (-not (Test-Path $src)) { Write-Log "skip $rel (absent locally)"; continue }
    $dst = Join-Path $Dest $rel

    robocopy $src $dst /E /Z /XO /R:1 /W:2 /MT:8 /NFL /NDL /NP /NJH /NJS | Out-Null
    $rc = $LASTEXITCODE
    # robocopy: 0-7 success, >=8 real failure
    if ($rc -ge 8) { $failed += "$rel (rc=$rc)"; Write-Log "FAILED  $rel  rc=$rc" }
    else           { Write-Log "ok      $rel  rc=$rc" }
}

if ($failed.Count -gt 0) {
    Write-Log "=== finished WITH ERRORS: $($failed -join ', ') ==="
    exit 1
}

Write-Log '=== cold push complete ==='
exit 0
