<#
.SYNOPSIS
    Deploys Simulation 14 (Thompson Sampling) to HPC.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [double[]]$Pvals = @(0.01, 0.005, 0.001),
    [double[]]$Separations = @(5, 6, 7)
)

$ErrorActionPreference = "Stop"

# Define Local Paths
$SimDir = "d:\GitHub\sparse_kmeans\simulations\sim14_thompson"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Create Remote Directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/sim14_thompson && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer Simulation Files (Driver, Submit Script)
Write-Host "Transferring simulation files..." -ForegroundColor Cyan
scp "${SimDir}\driver.R" "${SimDir}\submit.sh" "${Username}@${Hostname}:${RemoteBase}/simulations/sim14_thompson/"

# 3. Transfer Library Files
Write-Host "Transferring library files..." -ForegroundColor Cyan
scp "${CodeDir}\*.R" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Clean up, convert line endings, and Submit
Write-Host "Preparing HPC environment..." -ForegroundColor Cyan
$prepCmd = "cd ${RemoteBase}/simulations/sim14_thompson && rm -rf logs results_raw results_aggregated *.out *.err && mkdir -p logs results_raw results_aggregated && dos2unix *.sh *.R 2>/dev/null && chmod +x *.sh"
ssh "${Username}@${Hostname}" $prepCmd

Write-Host "Submitting jobs..." -ForegroundColor Cyan
$submittedJobs = @()

foreach ($sep in $Separations) {
    foreach ($pval in $Pvals) {
        Write-Host "Submitting simulation for sep: $sep | pval: $pval"
        
        $pvalStr = [string]$pval -replace '\.', 'p'
        $sbatchCmd = "cd ${RemoteBase}/simulations/sim14_thompson && sbatch --output=logs/sim_id%a_sep${sep}_pval${pvalStr}.out --error=logs/sim_id%a_sep${sep}_pval${pvalStr}.err --export=ALL,SEP=$sep,PVAL=$pval submit.sh"
        
        $output = ssh "${Username}@${Hostname}" $sbatchCmd 2>&1
        
        if ($LASTEXITCODE -eq 0 -and $output -match "Submitted batch job (\d+)") {
            $jobId = $matches[1]
            Write-Host " -> Successfully submitted Job ID: $jobId" -ForegroundColor Green
            $submittedJobs += [PSCustomObject]@{ Sep = $sep; Pval = $pval; JobID = $jobId; Status = "Success" }
        }
        else {
            Write-Error " -> Failed to submit job for sep $sep, pval $pval. Output: $output"
            $submittedJobs += [PSCustomObject]@{ Sep = $sep; Pval = $pval; JobID = "N/A"; Status = "Failed" }
        }
    }
}

Write-Host ""
Write-Host "================ Deployment Summary ================" -ForegroundColor Cyan
$submittedJobs | Format-Table -AutoSize
Write-Host "===================================================="
