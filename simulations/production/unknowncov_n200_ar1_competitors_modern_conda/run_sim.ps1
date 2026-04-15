# run_sim.ps1
# Launch unknowncov_ar1_conda simulations using the correct conda R environment.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File ".\run_sim.ps1" laplace
#   powershell -ExecutionPolicy Bypass -File ".\run_sim.ps1" gaussian
#   powershell -ExecutionPolicy Bypass -File ".\run_sim.ps1" both
#
param(
    [ValidateSet("laplace", "gaussian", "both", "")]
    [string]$Sim = ""
)

$CondaEnvDir = "C:\Users\jongmin\miniconda3\envs\r_legacy_sim"
$RscriptExe  = "$CondaEnvDir\Scripts\Rscript.exe"
$Root        = $PSScriptRoot

# Prepend conda env paths so Rscript.exe can find its DLLs (fixes 0xC0000135)
$OrigPath = $env:PATH
$env:PATH = "$CondaEnvDir;$CondaEnvDir\Scripts;$CondaEnvDir\Library\bin;$CondaEnvDir\Library\mingw-w64\bin;$CondaEnvDir\lib\R\bin\x64;$OrigPath"

function Invoke-Sim {
    param([string]$ScriptPath, [string]$Label)

    $ScriptName = Split-Path $ScriptPath -Leaf
    Write-Host "`n==> Running $Label simulation ($ScriptName)..." -ForegroundColor Cyan

    Push-Location $Root
    try {
        & $RscriptExe $ScriptName
    } finally {
        Pop-Location
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $Label simulation failed (exit code $LASTEXITCODE)." -ForegroundColor Red
    } else {
        Write-Host "==> $Label simulation completed." -ForegroundColor Green
    }
}

$SimType = if ([string]::IsNullOrWhiteSpace($Sim)) { "both" } else { $Sim.ToLower() }

switch ($SimType) {
    "laplace"  { 
        Invoke-Sim (Join-Path $Root "sim_laplace_ar1_conda.R") "Laplace" 
    }
    "gaussian" { 
        Invoke-Sim (Join-Path $Root "sim_gaussian_ar1_conda.R") "Gaussian" 
    }
    "both"     {
        Invoke-Sim (Join-Path $Root "sim_laplace_ar1_conda.R") "Laplace"
        Invoke-Sim (Join-Path $Root "sim_gaussian_ar1_conda.R") "Gaussian"
    }
    default    {
        Write-Host "Unknown simulation type: $Sim" -ForegroundColor Yellow
    }
}
