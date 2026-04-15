<#
.SYNOPSIS
    Deploys AR(1) Unknown-Covariance simulation to the HPC cluster.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [double[]]$Separations = @(4),
    [int[]]$Dimensions = @(1000, 2000, 3000, 4000, 5000),
    [string]$Noise = "Laplace"
)

$ErrorActionPreference = "Stop"

$SimName = Split-Path -Leaf $PSScriptRoot
$SimDir = $PSScriptRoot
$CodeDir = Join-Path $PSScriptRoot "..\..\..\code_r"

# 1. Create remote directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/production/${SimName} && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer simulation scripts
Write-Host "Transferring simulation scripts..." -ForegroundColor Cyan
scp "${SimDir}\*.R" "${SimDir}\*.sh" `
    "${Username}@${Hostname}:${RemoteBase}/simulations/production/${SimName}/"

# 3. Transfer R + C++ library sources
Write-Host "Transferring library dependencies..." -ForegroundColor Cyan
scp "${CodeDir}\*.R"   "${Username}@${Hostname}:${RemoteBase}/code_r/"
scp "${CodeDir}\*.cpp" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Prepare HPC environment
Write-Host "Preparing HPC environment..." -ForegroundColor Cyan
$prepCmd = "cd ${RemoteBase}/simulations/production/${SimName} && rm -rf logs results_raw results_aggregated *.out *.err && mkdir -p logs results_raw && dos2unix *.sh *.R 2>/dev/null; chmod +x *.sh"
ssh "${Username}@${Hostname}" $prepCmd

# 5. Compile C++ dependencies
Write-Host "Compiling C++ sources on HPC login node..." -ForegroundColor Cyan
$compileCmd = "module purge; module load gcc rstats/4.5.1 || module load r/4.5.1 || module load r; cd ${RemoteBase}/code_r && R CMD SHLIB selection_utils.cpp -o selection_utils.so && R CMD SHLIB proj_simplex.cpp -o proj_simplex.so"
ssh "${Username}@${Hostname}" $compileCmd
Write-Host "Compilation complete." -ForegroundColor Green

# 6. Submit array jobs
Write-Host "Submitting array jobs..." -ForegroundColor Cyan
$submittedJobs = @()

foreach ($sep in $Separations) {
    foreach ($p in $Dimensions) {
        Write-Host "  Submitting: sep=$sep, p=$p, noise=$Noise ..."

        $logPattern = "logs/sim_p${p}_${Noise}"
        $sbatchCmd = "cd ${RemoteBase}/simulations/production/${SimName} && sbatch --output=${logPattern}_%a.out --error=${logPattern}_%a.err --export=ALL,SEP=$sep,P=$p,NOISE=$Noise submit.sh"

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
                    Write-Host "    Retry $attempt/$maxRetries in ${retryWait}s..." -ForegroundColor Yellow
                    Start-Sleep -Seconds $retryWait
                }
            }
        }

        if ($success) {
            $jobId = $matches[1]
            Write-Host "    -> Array Job ID: $jobId" -ForegroundColor Green
            $submittedJobs += [PSCustomObject]@{
                Sep    = $sep
                P      = $p
                Noise  = $Noise
                JobID  = $jobId
                Status = "Success"
            }
        }
        else {
            Write-Error "    -> Failed after $maxRetries attempts. Output: $output"
            $submittedJobs += [PSCustomObject]@{
                Sep    = $sep
                P      = $p
                Noise  = $Noise
                JobID  = "N/A"
                Status = "Failed"
            }
        }

        # Brief pause to avoid SSH rate-limiting on head node
        Start-Sleep -Seconds 15
    }
}

Write-Host "`n================ Deployment Summary ================" -ForegroundColor Cyan
$submittedJobs | Format-Table -AutoSize
Write-Host "===================================================="
