<#
    restore_from_das.ps1  --  plan.md Phase 0.7 / R10.A

    Restores one dated evidence snapshot INTO A SCRATCH DIRECTORY so it can be
    compared against the live store. It will not write into C:\TradingBotData or
    the diagnostics tree - evidence_snapshot.restore() refuses those paths
    outright, because a drill that overwrites live state is how a drill becomes
    an incident.

    A backup nobody has ever restored is a hypothesis. Run this at least once
    against a real snapshot; the System Health tile reports when you last did.

    Dry run first (writes nothing, tells you what it would do):
      powershell -File C:\TradingBotData\_tools\restore_from_das.ps1 -Date 2026-08-22 -DryRun

    Then the real thing, into a scratch folder:
      powershell -File C:\TradingBotData\_tools\restore_from_das.ps1 -Date 2026-08-22 -Into D:\restore-test

    -Verify re-hashes every stored file against the snapshot's manifest and
    reports mismatches without restoring anything.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Date,
    [string] $Repo    = 'C:\Users\Aaron\TradingBotV3',
    [string] $Source  = '',
    [string] $Staging = "$env:LOCALAPPDATA\TradingBotV3\machine_cache\evidence_snapshots",
    [string] $Dest    = '\\MINI-PC\Trading Bot Data',
    [string] $Into,
    [switch] $DryRun,
    [switch] $Verify
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $Repo '.venv\Scripts\python.exe'
$module = Join-Path $Repo 'scripts\ops\evidence_snapshot.py'
if (-not (Test-Path $python)) { Write-Output "python not found: $python"; exit 1 }

# Prefer the local staging copy - it is the same bytes, on a faster disk, and it
# is present even when the share is not. Fall back to the DAS.
if (-not $Source) {
    $local = Join-Path $Staging $Date
    $remote = Join-Path (Join-Path $Dest 'backups') $Date
    if (Test-Path $local)       { $Source = $local;  Write-Output "using local staging copy: $local" }
    elseif (Test-Path $remote)  { $Source = $remote; Write-Output "using DAS copy: $remote" }
    else {
        Write-Output "no snapshot for $Date in either $local or $remote"
        exit 1
    }
}

if ($Verify) {
    & $python $module --verify $Source
    exit $LASTEXITCODE
}

if (-not $Into) { Write-Output 'a restore needs -Into <scratch directory>'; exit 2 }
$argsList = @($module, '--restore', $Source, '--into', $Into)
if ($DryRun) { $argsList += '--dry-run' }
& $python @argsList
exit $LASTEXITCODE
