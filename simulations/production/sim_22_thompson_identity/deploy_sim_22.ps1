<#
.SYNOPSIS
    Deploys Simulation 22 (Thompson Identity Data) to the High Performance Computing node.
    Follows established conventions mirroring `sim14` and `sim19` to map dynamic environments.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [double[]]$Separations = @(4),
    [int[]]$Dimensions = @(27000, 30000)
)

$ErrorActionPreference = "Stop"

# Define Local Paths tracking the architecture structure
$SimDir = "d:\GitHub\sparse_kmeans\simulations\production\sim_22_thompson_identity"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Create Remote Directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/production/sim_22_thompson_identity && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer Simulation Files (Driver, Submit Script)
Write-Host "Transferring simulation files..." -ForegroundColor Cyan
scp "${SimDir}\driver.R" "${SimDir}\submit.sh" "${Username}@${Hostname}:${RemoteBase}/simulations/production/sim_22_thompson_identity/"

# 3. Transfer Library Files and C++ source
Write-Host "Transferring library dependencies natively..." -ForegroundColor Cyan
scp "${CodeDir}\*.R" "${Username}@${Hostname}:${RemoteBase}/code_r/"
scp "${CodeDir}\*.cpp" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Clean up, convert line endings recursively, and Submit
Write-Host "Preparing HPC environment..." -ForegroundColor Cyan
$prepCmd = "cd ${RemoteBase}/simulations/production/sim_22_thompson_identity && rm -rf logs results_raw results_aggregated *.out *.err && mkdir -p logs results_raw results_aggregated && dos2unix *.sh *.R 2>/dev/null && chmod +x *.sh"
ssh "${Username}@${Hostname}" $prepCmd

# 5. Compile C++/Rcpp dependencies on the HPC
Write-Host "Compiling C++ source files on the HPC..." -ForegroundColor Cyan
# Ensure we load a valid compiler/R module before running R CMD SHLIB.
# Assuming standard HPC module setup (e.g. gcc and R)
$compileCmd = "module purge; module load gcc rstats/4.5.1 || module load rstats; cd ${RemoteBase}/code_r && R CMD SHLIB selection_utils.cpp -o selection_utils.so"
ssh "${Username}@${Hostname}" $compileCmd

Write-Host "Compilation complete. Shared object selection_utils.so should be ready." -ForegroundColor Green

# 6. Submit simulation array jobs to the Slurm scheduler
Write-Host "Submitting simulation array jobs to the Slurm scheduler..." -ForegroundColor Cyan
$submittedJobs = @()

foreach ($sep in $Separations) {
    foreach ($p in $Dimensions) {
        Write-Host "Submitting simulation for sep: $sep, p: $p..."
        
        # Export SEP and P to the submit.sh script automatically 
        $sbatchCmd = "cd ${RemoteBase}/simulations/production/sim_22_thompson_identity && sbatch --output=logs/sim_id%a_sep${sep}_p${p}.out --error=logs/sim_id%a_sep${sep}_p${p}.err --export=ALL,SEP=$sep,P=$p submit.sh"
        
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
            $submittedJobs += [PSCustomObject]@{ Sep = $sep; P = $p; JobID = $jobId; Status = "Success" }
        }
        else {
            Write-Error " -> Failed to submit job for separation condition $sep, dimension $p after $maxRetries attempts. Output: $output"
            $submittedJobs += [PSCustomObject]@{ Sep = $sep; P = $p; JobID = "N/A"; Status = "Failed" }
        }
        
        # Add a brief pause to prevent SSH connection rate limiting/timeouts on the head node
        Start-Sleep -Seconds 5
    }
}

Write-Host "`n================ Deployment Summary ================" -ForegroundColor Cyan
$submittedJobs | Format-Table -AutoSize
Write-Host "===================================================="
