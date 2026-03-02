<#
.SYNOPSIS
    Deploys Simulation 15 (Witten and Arias-Castro Sparse K-Means) to HPC.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [double[]]$SepTargets = @(4, 5)
)

$ErrorActionPreference = "Stop"

# Define Local Paths
$SimDir = "d:\GitHub\sparse_kmeans\simulations\sim_15_witten_arias"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Create Remote Directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/sim_15_witten_arias && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer Simulation Files (Driver, Submit Script)
Write-Host "Transferring simulation files..." -ForegroundColor Cyan
scp "${SimDir}\driver.R" "${SimDir}\submit.sh" "${Username}@${Hostname}:${RemoteBase}/simulations/sim_15_witten_arias/"

# 3. Transfer Library Files
Write-Host "Transferring library files..." -ForegroundColor Cyan
scp "${CodeDir}\*.R" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Clean up, convert line endings, and Submit
Write-Host "Preparing HPC environment..." -ForegroundColor Cyan
$prepCmd = "cd ${RemoteBase}/simulations/sim_15_witten_arias && rm -rf logs results *.out *.err && mkdir -p logs results && dos2unix *.sh *.R 2>/dev/null && chmod +x *.sh"
ssh "${Username}@${Hostname}" $prepCmd

Write-Host "Submitting jobs..." -ForegroundColor Cyan
$submittedJobs = @()

foreach ($sep_val in $SepTargets) {
    Write-Host "Submitting simulation for Separation: $sep_val"
    
    $sbatchCmd = "cd ${RemoteBase}/simulations/sim_15_witten_arias && sbatch submit.sh $sep_val"
    
    $output = ssh "${Username}@${Hostname}" $sbatchCmd 2>&1
    
    if ($LASTEXITCODE -eq 0 -and $output -match "Submitted batch job (\d+)") {
        $jobId = $matches[1]
        Write-Host " -> Successfully submitted Job ID: $jobId" -ForegroundColor Green
        $submittedJobs += [PSCustomObject]@{ Sep_Val = $sep_val; JobID = $jobId; Status = "Success" }
    }
    else {
        Write-Error " -> Failed to submit job for Sep_val $sep_val. Output: $output"
        $submittedJobs += [PSCustomObject]@{ Sep_Val = $sep_val; JobID = "N/A"; Status = "Failed" }
    }
}

Write-Host ""
Write-Host "================ Deployment Summary ================" -ForegroundColor Cyan
$submittedJobs | Format-Table -AutoSize
Write-Host "===================================================="
