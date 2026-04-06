<#
.SYNOPSIS
    Deploys Thompson Unknown-Covariance Naive to the HPC cluster.
    Mirrors the sim_22 structure mapping deployment logic directly.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [double[]]$Separations = @(6),
    [int[]]$Dimensions = @(1500, 2000),
    [string]$Noise = "Gaussian"
)

$ErrorActionPreference = "Stop"

# Define Local Paths tracking the architecture structure
$SimName = "thompson_unknowncov_naive"
$SimDir = "d:\GitHub\sparse_kmeans\simulations\production\$SimName"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Create Remote Directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/production/${SimName} && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer Simulation Files (Driver, Submit Script)
Write-Host "Transferring simulation files..." -ForegroundColor Cyan
scp "${SimDir}\driver.R" "${SimDir}\submit.sh" "${Username}@${Hostname}:${RemoteBase}/simulations/production/${SimName}/"

# 3. Transfer Library Files and C++ source
Write-Host "Transferring library dependencies natively..." -ForegroundColor Cyan
scp "${CodeDir}\*.R" "${Username}@${Hostname}:${RemoteBase}/code_r/"
scp "${CodeDir}\*.cpp" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Clean up, convert line endings recursively, and Submit
Write-Host "Preparing HPC environment..." -ForegroundColor Cyan
$prepCmd = "cd ${RemoteBase}/simulations/production/${SimName} && rm -rf logs results_raw results_aggregated *.out *.err && mkdir -p logs results_raw results_aggregated && dos2unix *.sh *.R 2>/dev/null && chmod +x *.sh"
ssh "${Username}@${Hostname}" $prepCmd

# 5. Compile C++/Rcpp dependencies on the HPC
Write-Host "Compiling C++ source files on the HPC..." -ForegroundColor Cyan
$compileCmd = "module purge; module load gcc rstats/4.5.1 || module load r/4.5.1 || module load r; cd ${RemoteBase}/code_r && R CMD SHLIB selection_utils.cpp -o selection_utils.so && R CMD SHLIB proj_simplex.cpp -o proj_simplex.so"
ssh "${Username}@${Hostname}" $compileCmd
Write-Host "Compilation complete." -ForegroundColor Green

# 6. Submit simulation array jobs to the Slurm scheduler
Write-Host "Submitting simulation array jobs to the Slurm scheduler..." -ForegroundColor Cyan
$submittedJobs = @()

foreach ($sep in $Separations) {
    foreach ($p in $Dimensions) {
        Write-Host "Submitting simulation for sep: $sep, p: $p, noise: $Noise..."
        
        $logPattern = "logs/sim_id%a_sep${sep}_p${p}_${Noise}"
        $sbatchCmd = "cd ${RemoteBase}/simulations/production/${SimName} && sbatch --output=${logPattern}.out --error=${logPattern}.err --export=ALL,SEP=$sep,P=$p,NOISE=$Noise submit.sh"
        
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
            $submittedJobs += [PSCustomObject]@{ Sep = $sep; P = $p; Noise = $Noise; JobID = $jobId; Status = "Success" }
        }
        else {
            Write-Error " -> Failed to submit job for separation condition $sep, dimension $p after $maxRetries attempts. Output: $output"
            $submittedJobs += [PSCustomObject]@{ Sep = $sep; P = $p; Noise = $Noise; JobID = "N/A"; Status = "Failed" }
        }
        
        # Add a brief pause to prevent SSH connection rate limiting/timeouts on the head node
        Start-Sleep -Seconds 15
    }
}

Write-Host "`n================ Deployment Summary ================" -ForegroundColor Cyan
$submittedJobs | Format-Table -AutoSize
Write-Host "===================================================="
