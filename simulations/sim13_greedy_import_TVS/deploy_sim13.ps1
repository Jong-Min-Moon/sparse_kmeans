
<#
.SYNOPSIS
    Deploys Simulation 07 (Permutation FDR 0.4) to HPC.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [double[]]$FdrTargets = @(0.4, 0.5, 0.6)
)

$ErrorActionPreference = "Stop"

# Define Local Paths
$SimDir = "d:\GitHub\sparse_kmeans\simulations\sim13_greedy_import_TVS"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Create Remote Directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/sim13_greedy_import_TVS && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer Simulation Files (Driver, Submit Script)
Write-Host "Transferring simulation files..." -ForegroundColor Cyan
scp "${SimDir}\driver.R" "${SimDir}\submit.sh" "${Username}@${Hostname}:${RemoteBase}/simulations/sim13_greedy_import_TVS/"

# 3. Transfer Library Files
Write-Host "Transferring library files..." -ForegroundColor Cyan
# Crucial: Ensure the new block_coordinate_optim_greedy_unknowncov_SAM.R is transferred
scp "${CodeDir}\*.R" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Clean up, convert line endings, and Submit
Write-Host "Preparing HPC environment..." -ForegroundColor Cyan
$prepCmd = "cd ${RemoteBase}/simulations/sim13_greedy_import_TVS && rm -rf logs results *.out *.err && mkdir -p logs results && dos2unix *.sh *.R 2>/dev/null && chmod +x *.sh"
ssh "${Username}@${Hostname}" $prepCmd

Write-Host "Submitting jobs..." -ForegroundColor Cyan
$submittedJobs = @()

foreach ($fdr in $FdrTargets) {
    Write-Host "Submitting simulation for FDR Target: $fdr"
    
    $fdrStr = [string]$fdr -replace '\.', 'p'
    $sbatchCmd = "cd ${RemoteBase}/simulations/sim13_greedy_import_TVS && sbatch --output=logs/sim_id%a_fdr$fdrStr.out --error=logs/sim_id%a_fdr$fdrStr.err --export=ALL,FDR_TARGET=$fdr submit.sh"
    
    $output = ssh "${Username}@${Hostname}" $sbatchCmd 2>&1
    
    if ($LASTEXITCODE -eq 0 -and $output -match "Submitted batch job (\d+)") {
        $jobId = $matches[1]
        Write-Host " -> Successfully submitted Job ID: $jobId" -ForegroundColor Green
        $submittedJobs += [PSCustomObject]@{ FDR = $fdr; JobID = $jobId; Status = "Success" }
    }
    else {
        Write-Error " -> Failed to submit job for FDR $fdr. Output: $output"
        $submittedJobs += [PSCustomObject]@{ FDR = $fdr; JobID = "N/A"; Status = "Failed" }
    }
}

Write-Host ""
Write-Host "================ Deployment Summary ================" -ForegroundColor Cyan
$submittedJobs | Format-Table -AutoSize
Write-Host "===================================================="
