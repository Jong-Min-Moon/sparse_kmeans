
<#
.SYNOPSIS
    Deploys Simulation 01 (Greedy Varying P) to HPC.

.DESCRIPTION
    1. Copies 'sim01_greedy_varying_p' folder to ~/sparse_kmeans_project/simulations/sim01_greedy_varying_p
    2. Copies 'code_r' folder to ~/sparse_kmeans_project/code_r
    3. Runs sbatch submit.sh

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
$SimDir = "d:\GitHub\sparse_kmeans\simulations\sim01_greedy_varying_p"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Create Remote Directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
# SSH command to create directories (-p ensures no error if exists)
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/sim01_greedy_varying_p && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer Simulation Files
Write-Host "Transferring simulation files..." -ForegroundColor Cyan
scp -r "${SimDir}\*" "${Username}@${Hostname}:${RemoteBase}/simulations/sim01_greedy_varying_p/"

# 3. Transfer Library Files
Write-Host "Transferring library files..." -ForegroundColor Cyan
scp -r "${CodeDir}\*" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Convert Line Endings & Submit
Write-Host "Submitting multi-parameter job suite..." -ForegroundColor Cyan
# Convert endings for R scripts and shell scripts, then run submit_all.sh
$submitCmd = "cd ${RemoteBase}/simulations/sim01_greedy_varying_p && dos2unix submit.sh compile_solver.R run_sim01.R submit_all.sh submit_template.sh && chmod +x submit_all.sh && ./submit_all.sh"


ssh "${Username}@${Hostname}" $submitCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Job submitted successfully!" -ForegroundColor Green
}
else {
    Write-Error "Job submission failed."
}
