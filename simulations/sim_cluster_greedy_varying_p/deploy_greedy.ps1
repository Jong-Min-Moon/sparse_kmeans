<#
.SYNOPSIS
    Deploys Simulation for cluster_greedy (Varying p) to the HPC.
    Follows protocol from `sim_22_thompson_identity`.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [int[]]$Dimensions = @(9000, 10000),
    [double]$FDR = 0.4
)

$ErrorActionPreference = "Stop"

# Define Local Paths
$SimDir = "d:\GitHub\sparse_kmeans\simulations\sim_cluster_greedy_varying_p"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Create Remote Directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/sim_cluster_greedy_varying_p && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer Simulation Files
Write-Host "Transferring simulation files..." -ForegroundColor Cyan
scp "${SimDir}\driver_greedy.R" "${SimDir}\submit_greedy.sh" "${SimDir}\aggregate_greedy.R" "${Username}@${Hostname}:${RemoteBase}/simulations/sim_cluster_greedy_varying_p/"

# 3. Transfer Library Files and C++ source
Write-Host "Transferring library dependencies..." -ForegroundColor Cyan
scp "${CodeDir}\*.R" "${Username}@${Hostname}:${RemoteBase}/code_r/"
scp "${CodeDir}\*.cpp" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Prepare HPC environment
Write-Host "Preparing HPC environment..." -ForegroundColor Cyan
$prepCmd = "cd ${RemoteBase}/simulations/sim_cluster_greedy_varying_p && rm -rf logs results_raw results_aggregated *.out *.err && mkdir -p logs results_raw results_aggregated && dos2unix *.sh *.R 2>/dev/null && chmod +x *.sh"
ssh "${Username}@${Hostname}" $prepCmd

# 5. Compile C++/Rcpp dependencies on the HPC (MANDATORY STEP)
Write-Host "Compiling C++ source files on the HPC..." -ForegroundColor Cyan
# Using the same module and command as sim_22 protocol
$compileCmd = "module purge; module load gcc rstats/4.5.1 || module load rstats; cd ${RemoteBase}/code_r && R CMD SHLIB selection_utils.cpp -o selection_utils.so"
ssh "${Username}@${Hostname}" $compileCmd

Write-Host "Compilation complete. Shared object selection_utils.so should be ready on HPC." -ForegroundColor Green

# 6. Submit simulation array jobs
Write-Host "Submitting simulation array jobs to Slurm..." -ForegroundColor Cyan
$submittedJobs = @()

foreach ($p in $Dimensions) {
    Write-Host "Submitting simulation for p: $p, FDR: $FDR..."
    
    # Export P and FDR matching submit_greedy.sh variables
    $sbatchCmd = "cd ${RemoteBase}/simulations/sim_cluster_greedy_varying_p && sbatch --output=logs/sim_id%a_p${p}.out --error=logs/sim_id%a_p${p}.err --export=ALL,P=$p,FDR=$FDR submit_greedy.sh"
    
    $maxRetries = 5
    $retryWait = 5
    $attempt = 0
    $success = $false
    $output = ""
    
    while ($attempt -lt $maxRetries -and -not $success) {
        $output = ssh "${Username}@${Hostname}" $sbatchCmd 2>&1
        if ($LASTEXITCODE -eq 0 -and $output -match "Submitted batch job (\d+)") {
            $success = $true
        }
        else {
            $attempt++
            if ($attempt -lt $maxRetries) {
                Write-Host "SSH timeout or cluster rate limit hit. Retrying $attempt/$maxRetries in $retryWait seconds..." -ForegroundColor Yellow
                Start-Sleep -Seconds $retryWait
            }
        }
    }
    
    if ($success) {
        $jobId = $matches[1]
        Write-Host " -> Successfully submitted Array Job ID: $jobId" -ForegroundColor Green
        $submittedJobs += [PSCustomObject]@{ P = $p; JobID = $jobId; Status = "Success" }
    }
    else {
        Write-Error " -> Failed to submit job for dimension $p after $maxRetries attempts. Output: $output"
        $submittedJobs += [PSCustomObject]@{ P = $p; JobID = "N/A"; Status = "Failed" }
    }
    
    Start-Sleep -Seconds 10
}

Write-Host "`n================ Deployment Summary ================" -ForegroundColor Cyan
$submittedJobs | Format-Table -AutoSize
Write-Host "===================================================="
