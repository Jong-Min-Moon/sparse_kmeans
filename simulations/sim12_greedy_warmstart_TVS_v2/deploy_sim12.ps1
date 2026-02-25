
<#
.SYNOPSIS
    Deploys Simulation 07 (Permutation FDR 0.4) to HPC.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$ErrorActionPreference = "Stop"

# Define Local Paths
$SimDir = "d:\GitHub\sparse_kmeans\simulations\sim12_greedy_warmstart_TVS_v2"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Create Remote Directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/sim12_greedy_warmstart_TVS_v2 && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer Simulation Files (Driver, Submit Script)
Write-Host "Transferring simulation files..." -ForegroundColor Cyan
scp "${SimDir}\driver.R" "${SimDir}\submit.sh" "${Username}@${Hostname}:${RemoteBase}/simulations/sim12_greedy_warmstart_TVS_v2/"

# 3. Transfer Library Files
Write-Host "Transferring library files..." -ForegroundColor Cyan
# Crucial: Ensure the new block_coordinate_optim_permutation.R is transferred
scp "${CodeDir}\*.R" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Convert Line Endings & Submit
Write-Host "Submitting job..." -ForegroundColor Cyan
$submitCmd = "cd ${RemoteBase}/simulations/sim12_greedy_warmstart_TVS_v2 && rm -rf logs results *.out *.err && dos2unix *.sh *.R && chmod +x *.sh && sbatch submit.sh"
ssh "${Username}@${Hostname}" $submitCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Job submitted successfully!" -ForegroundColor Green
}
else {
    Write-Error "Job submission failed."
}
