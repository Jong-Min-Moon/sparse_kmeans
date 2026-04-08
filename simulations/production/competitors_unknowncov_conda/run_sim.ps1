# run_sim.ps1
# Launch unknowncov_conda simulations using the correct conda R environment.
#
# Usage — must invoke with ExecutionPolicy Bypass since script signing is off:
#   powershell -ExecutionPolicy Bypass -File ".\run_sim.ps1" laplace
#   powershell -ExecutionPolicy Bypass -File ".\run_sim.ps1" gaussian
#   powershell -ExecutionPolicy Bypass -File ".\run_sim.ps1"   # both
#
# param() MUST be the first non-comment statement in a .ps1 file.
param(
    [ValidateSet("laplace", "gaussian", "both", "")]
    [string]$Sim = ""
)

$CondaEnv = "r_legacy_sim"
$Root     = $PSScriptRoot   # always the directory containing this script

function Invoke-Sim {
    param([string]$Script, [string]$Label)
    Write-Host "`n==> Running $Label simulation..." -ForegroundColor Cyan
    # --no-capture-output streams R's cat() output live to this console
    conda run --no-capture-output -n $CondaEnv Rscript "$Script"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $Label simulation failed (exit code $LASTEXITCODE)." -ForegroundColor Red
    } else {
        Write-Host "==> $Label simulation completed." -ForegroundColor Green
    }
}

switch ($Sim.ToLower()) {
    "laplace"  { Invoke-Sim "$Root\sim_laplace_unknowncov.R"  "Laplace" }
    "gaussian" { Invoke-Sim "$Root\sim_gaussian_unknowncov.R" "Gaussian" }
    default    {
        Invoke-Sim "$Root\sim_laplace_unknowncov.R"  "Laplace"
        Invoke-Sim "$Root\sim_gaussian_unknowncov.R" "Gaussian"
    }
}
