
<#
.SYNOPSIS
    Deploys Simulation 02 (Single Run, Unknown Covariance) to HPC.

.DESCRIPTION
    1. Copies the 'sim02_unknown_cov_greedy_single' folder to ~/sparse_kmeans_project/simulations/sim02_unknown_cov_greedy_single
    2. Copies the 'code_r' folder to ~/sparse_kmeans_project/code_r (to ensure dependencies are up to date)
    3. Runs submit_all.sh

.PARAMETER Username
    default: jongminm

.PARAMETER Hostname
    default: discovery.usc.edu

.PARAMETER RemoteBase
    default: ~/sparse_kmeans_project
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$ErrorActionPreference = "Stop"

# Define Local Paths
$SimDir = "d:\GitHub\sparse_kmeans\simulations\sim02_unknown_cov_greedy_single"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Create Remote Directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/sim02_unknown_cov_greedy_single && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer Simulation Files
Write-Host "Transferring simulation files..." -ForegroundColor Cyan
scp -r "${SimDir}\*" "${Username}@${Hostname}:${RemoteBase}/simulations/sim02_unknown_cov_greedy_single/"

# 3. Transfer Library Files
Write-Host "Transferring library files..." -ForegroundColor Cyan
scp -r "${CodeDir}\*" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Convert Line Endings & Submit
Write-Host "Submitting jobs..." -ForegroundColor Cyan
$submitCmd = "cd ${RemoteBase}/simulations/sim02_unknown_cov_greedy_single && dos2unix submit.sh submit_all.sh driver.R && bash submit_all.sh"

ssh "${Username}@${Hostname}" $submitCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Job submitted successfully!" -ForegroundColor Green
}
else {
    Write-Error "Job submission failed."
}
