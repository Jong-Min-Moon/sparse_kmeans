<#
.SYNOPSIS
    Deploys Simulation 19 (Greedy Naive Bayes) to HPC.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [double[]]$Separations = @(4, 5, 6)
)

$ErrorActionPreference = "Stop"

# Define Local Paths
# Detect where script runs from, matching environment natively
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SimDir = $ScriptDir
$CodeDir = Join-Path (Split-Path (Split-Path $SimDir -Parent) -Parent) "code_r"

# 1. Create Remote Directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/sim_19_greedy_naive_bayes && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer Simulation Files (Driver, Submit Script)
Write-Host "Transferring simulation files..." -ForegroundColor Cyan
scp "${SimDir}\simulate_sim_19.R" "${SimDir}\submit.sh" "${Username}@${Hostname}:${RemoteBase}/simulations/sim_19_greedy_naive_bayes/"

# 3. Transfer Library Files
Write-Host "Transferring library files..." -ForegroundColor Cyan
scp "${CodeDir}\*.R" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Clean up, convert line endings, and Submit
Write-Host "Preparing HPC environment..." -ForegroundColor Cyan
$prepCmd = "cd ${RemoteBase}/simulations/sim_19_greedy_naive_bayes && rm -rf logs results_raw results_aggregated slurm*.out slurm*.err && mkdir -p logs/sim_19_greedy_naive_bayes results_raw results_aggregated && dos2unix *.sh *.R 2>/dev/null && chmod +x *.sh"
ssh "${Username}@${Hostname}" $prepCmd

Write-Host "Submitting SLURM jobs..." -ForegroundColor Cyan
$submittedJobs = @()

foreach ($sep in $Separations) {
    Write-Host "Submitting simulation for sep: $sep | Target FDR: 0.4"
    
    $sbatchCmd = "cd ${RemoteBase}/simulations/sim_19_greedy_naive_bayes && sbatch --output=logs/sim_19_greedy_naive_bayes/sim_id%a_sep${sep}.out --error=logs/sim_19_greedy_naive_bayes/sim_id%a_sep${sep}.err --export=ALL,SEP=$sep submit.sh"
    
    $output = ssh "${Username}@${Hostname}" $sbatchCmd 2>&1
    
    if ($LASTEXITCODE -eq 0 -and $output -match "Submitted batch job (\d+)") {
        $jobId = $matches[1]
        Write-Host " -> Successfully submitted Job ID: $jobId" -ForegroundColor Green
        $submittedJobs += [PSCustomObject]@{ Sep = $sep; JobID = $jobId; Status = "Success" }
    }
    else {
        Write-Error " -> Failed to submit job for sep $sep. Output: $output"
        $submittedJobs += [PSCustomObject]@{ Sep = $sep; JobID = "N/A"; Status = "Failed" }
    }
}


Write-Host ""
Write-Host "================ Deployment Summary ================" -ForegroundColor Cyan
$submittedJobs | Format-Table -AutoSize
Write-Host "===================================================="
