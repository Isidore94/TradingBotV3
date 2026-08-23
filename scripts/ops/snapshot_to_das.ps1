<#
    snapshot_to_das.ps1  --  plan.md Phase 0.7 / R10.A

    TWO JOBS, TWO SCOPES. DO NOT MERGE THEM.

      push_cold_to_das.ps1  mirrors the COLD, append-only subtrees
      (data\daily_bars, data\intraday_bars, output, logs, away_report_archive,
      and the R10 month-segmented ledger dirs) to the DAS hourly and
      incrementally. Nothing there is dated and nothing is ever deleted.

      THIS takes a DATED SNAPSHOT of the HOT state that one deliberately
      excludes: data\runtime (~3.5 GB - the 960 MB setup tracker, the 203 MB
      outcome CSV, the journal SQLite, every outcome / cohort / focus store),
      the home-root evidence files, _tools, and the machine-local diagnostics
      tree (~529 MB). Those are rewritten constantly, so decision 0015 stands
      and they stay on the local SSD: this COPIES, it never moves.

    Trader, 2026-08-22: "Any and all very important files that we use
    occasionally should go to the server with the massive HDD."

    Local staging FIRST, DAS second - so a share that is unreachable exits 0
    and leaves a complete local snapshot behind. That is the intended fallback,
    exactly as it is for the cold push, and not an error.

    The real work is scripts\ops\evidence_snapshot.py, which is unit-tested;
    this file is the Task Scheduler entry point and the robocopy leg.

    Nightly, AFTER the AI runner:
      powershell -NoProfile -ExecutionPolicy Bypass -File C:\TradingBotData\_tools\snapshot_to_das.ps1
#>

[CmdletBinding()]
param(
    [string] $Repo       = 'C:\Users\Aaron\TradingBotV3',
    [string] $Dest       = '\\MINI-PC\Trading Bot Data',
    [string] $Staging    = "$env:LOCALAPPDATA\TradingBotV3\machine_cache\evidence_snapshots",
    [string] $SnapshotDate,
    [switch] $NoPrune,
    [switch] $StageOnly
)

$ErrorActionPreference = 'Stop'
$logDir = 'C:\TradingBotData\_tools\logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$stamp   = Get-Date -Format 'yyyyMMdd-HHmmss'
$logFile = Join-Path $logDir "snapshot-$stamp.log"

function Write-Log {
    param([string] $Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Message"
    Write-Output $line
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

Write-Log '=== evidence snapshot starting ==='
$python = Join-Path $Repo '.venv\Scripts\python.exe'
$module = Join-Path $Repo 'scripts\ops\evidence_snapshot.py'
if (-not (Test-Path $python)) { Write-Log "python not found: $python"; exit 1 }
if (-not (Test-Path $module)) { Write-Log "module not found: $module"; exit 1 }

# ---- stage locally -------------------------------------------------------
# This is the part that must succeed. A staged snapshot with no DAS is a
# working backup on the wrong disk; a DAS copy with no staging is nothing.
$argsList = @($module, '--staging', $Staging)
if ($SnapshotDate) { $argsList += @('--date', $SnapshotDate) }
if (-not $NoPrune)  { $argsList += '--prune' }

& $python @argsList 2>&1 | ForEach-Object { Write-Log $_ }
if ($LASTEXITCODE -ne 0) { Write-Log "staging FAILED (exit $LASTEXITCODE)"; exit 1 }

$day = if ($SnapshotDate) { $SnapshotDate } else { Get-Date -Format 'yyyy-MM-dd' }
$snapDir = Join-Path $Staging $day
if (-not (Test-Path $snapDir)) { Write-Log "staged snapshot missing: $snapDir"; exit 1 }
$bytes = (Get-ChildItem $snapDir -Recurse -File -Force -ErrorAction SilentlyContinue |
          Measure-Object -Property Length -Sum).Sum
Write-Log "staged $([math]::Round($bytes / 1MB, 1)) MB at $snapDir"
if ($StageOnly) { Write-Log 'StageOnly set - not copying to the DAS.'; exit 0 }

# ---- preflight the DAS ---------------------------------------------------
if (-not (Test-Path $Dest)) {
    Write-Log "MINI-PC UNREACHABLE - the snapshot stays staged locally. Not an error; retries next run."
    exit 0
}
$probe = Join-Path $Dest ".snapshot_probe_$stamp"
try {
    Set-Content -Path $probe -Value 'probe' -Encoding utf8 -ErrorAction Stop
    Remove-Item $probe -Force -ErrorAction SilentlyContinue
} catch {
    Write-Log "MINI-PC REACHABLE BUT NOT WRITABLE ($($_.Exception.Message.Trim())). Snapshot stays staged."
    exit 0
}

# ---- copy ----------------------------------------------------------------
# No /MIR: the DAS keeps what it already has. Retention is decided by the
# Python side against the staging root, never by mirroring a deletion across.
$dst = Join-Path (Join-Path $Dest 'backups') $day
robocopy $snapDir $dst /E /Z /R:1 /W:2 /MT:8 /NFL /NDL /NP /NJH /NJS | Out-Null
$rc = $LASTEXITCODE
if ($rc -ge 8) {
    Write-Log "robocopy FAILED rc=$rc - the staged snapshot is intact at $snapDir"
    exit 1
}
Write-Log "copied to $dst (rc=$rc)"
Write-Log '=== evidence snapshot complete ==='
exit 0
