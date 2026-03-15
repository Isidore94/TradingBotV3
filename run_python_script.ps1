param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ScriptPath,

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$requirementsFile = Join-Path $repoRoot "requirements.txt"
$requirementsStamp = Join-Path $repoRoot ".venv\requirements.sha256"

function Get-BootstrapPython {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        return @($pyLauncher.Source, "-3")
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return @($pythonCmd.Source)
    }

    throw "Python was not found on PATH. Install Python 3 and try again."
}

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$CommandParts
    )

    $exe = $CommandParts[0]
    $args = @()
    if ($CommandParts.Length -gt 1) {
        $args = $CommandParts[1..($CommandParts.Length - 1)]
    }

    & $exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($CommandParts -join ' ')"
    }
}

function Ensure-Venv {
    if (Test-Path $venvPython) {
        return
    }

    Write-Host "Creating project virtual environment in .venv..."
    $bootstrap = Get-BootstrapPython
    Invoke-ExternalCommand ($bootstrap + @("-m", "venv", ".venv"))
}

function Ensure-Requirements {
    if (-not (Test-Path $requirementsFile)) {
        return
    }

    $currentHash = (Get-FileHash -Algorithm SHA256 $requirementsFile).Hash
    $savedHash = ""
    if (Test-Path $requirementsStamp) {
        $savedHash = (Get-Content $requirementsStamp -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    }

    if ($currentHash -eq $savedHash) {
        return
    }

    Write-Host "Installing Python dependencies from requirements.txt..."
    Invoke-ExternalCommand @($venvPython, "-m", "pip", "install", "--disable-pip-version-check", "-r", $requirementsFile)
    Set-Content -Path $requirementsStamp -Value $currentHash -Encoding ASCII
}

$resolvedScriptPath = Join-Path $repoRoot $ScriptPath
if (-not (Test-Path $resolvedScriptPath)) {
    throw "Script not found: $resolvedScriptPath"
}

Ensure-Venv
Ensure-Requirements

& $venvPython $resolvedScriptPath @ScriptArgs
exit $LASTEXITCODE
