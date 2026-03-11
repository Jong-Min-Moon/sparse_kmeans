<#
.SYNOPSIS
    Deploys Simulation 22 (Thompson Identity Data) to the High Performance Computing node.
    Follows established conventions mirroring `sim14` and `sim19` to map dynamic environments.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [double[]]$Separations = @(4)
)

$ErrorActionPreference = "Stop"

# Define Local Paths tracking the architecture structure
$SimDir = "d:\GitHub\sparse_kmeans\simulations\sim_22_thompson_identity"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Create Remote Directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/sim_22_thompson_identity && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer Simulation Files (Driver, Submit Script)
Write-Host "Transferring simulation files..." -ForegroundColor Cyan
scp "${SimDir}\driver.R" "${SimDir}\submit.sh" "${Username}@${Hostname}:${RemoteBase}/simulations/sim_22_thompson_identity/"

# 3. Transfer Library Files
Write-Host "Transferring library dependencies natively..." -ForegroundColor Cyan
scp "${CodeDir}\*.R" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Clean up, convert line endings recursively, and Submit
Write-Host "Preparing HPC environment..." -ForegroundColor Cyan
$prepCmd = "cd ${RemoteBase}/simulations/sim_22_thompson_identity && rm -rf logs results_raw results_aggregated *.out *.err && mkdir -p logs results_raw results_aggregated && dos2unix *.sh *.R 2>/dev/null && chmod +x *.sh"
ssh "${Username}@${Hostname}" $prepCmd

Write-Host "Submitting simulation array jobs to the Slurm scheduler..." -ForegroundColor Cyan
$submittedJobs = @()

foreach ($sep in $Separations) {
    Write-Host "Submitting simulation for sep: $sep..."
    
    # Export SEP to the submit.sh script automatically 
    $sbatchCmd = "cd ${RemoteBase}/simulations/sim_22_thompson_identity && sbatch --output=logs/sim_id%a_sep${sep}.out --error=logs/sim_id%a_sep${sep}.err --export=ALL,SEP=$sep submit.sh"
    
    $output = ssh "${Username}@${Hostname}" $sbatchCmd 2>&1
    
    if ($LASTEXITCODE -eq 0 -and $output -match "Submitted batch job (\d+)") {
        $jobId = $matches[1]
        Write-Host " -> Successfully submitted Array Job ID: $jobId" -ForegroundColor Green
        $submittedJobs += [PSCustomObject]@{ Sep = $sep; JobID = $jobId; Status = "Success" }
    }
    else {
        Write-Error " -> Failed to submit job for separation condition $sep. Output: $output"
        $submittedJobs += [PSCustomObject]@{ Sep = $sep; JobID = "N/A"; Status = "Failed" }
    }
}

Write-Host "`n================ Deployment Summary ================" -ForegroundColor Cyan
$submittedJobs | Format-Table -AutoSize
Write-Host "===================================================="
