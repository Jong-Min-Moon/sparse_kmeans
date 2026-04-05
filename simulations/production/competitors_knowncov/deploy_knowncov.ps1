<#
.SYNOPSIS
    Deploys Competitors Known-Covariance simulation to the HPC cluster.
    Mirror the structure of sim_22.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [double[]]$Separations = @(4),
    [int[]]$Dimensions = @(9000, 15000, 21000, 27000),
    [string]$Noise = "Laplace",
    [string]$Methods = "arias"
)

$ErrorActionPreference = "Stop"

$SimName = "competitors_knowncov"
$SimDir = "d:\GitHub\sparse_kmeans\simulations\production\$SimName"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Create remote directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/production/${SimName} && mkdir -p ${RemoteBase}/simulations/production/competitors_unknowncov && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer simulation scripts
Write-Host "Transferring simulation scripts..." -ForegroundColor Cyan
scp "${SimDir}\hpc_driver.R" "${SimDir}\submit.sh" `
    "${Username}@${Hostname}:${RemoteBase}/simulations/production/${SimName}/"

# 3. Transfer R + C++ library sources
Write-Host "Transferring library dependencies..." -ForegroundColor Cyan
scp "${CodeDir}\*.R"   "${Username}@${Hostname}:${RemoteBase}/code_r/"
scp "${CodeDir}\*.cpp" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Transfer sim_utils.R (dependency)
Write-Host "Transferring unknowncov shared utilities..." -ForegroundColor Cyan
scp "d:\GitHub\sparse_kmeans\simulations\production\competitors_unknowncov\sim_utils.R" `
    "${Username}@${Hostname}:${RemoteBase}/simulations/production/competitors_unknowncov/"

# 5. Prepare HPC environment
Write-Host "Preparing HPC environment..." -ForegroundColor Cyan
$prepCmd = "cd ${RemoteBase}/simulations/production/${SimName} && rm -rf logs results_raw results_aggregated *.out *.err && mkdir -p logs results_raw && dos2unix *.sh *.R 2>/dev/null; chmod +x *.sh"
ssh "${Username}@${Hostname}" $prepCmd

# 6. Compile C++ dependencies
Write-Host "Compiling C++ sources on HPC login node..." -ForegroundColor Cyan
$compileCmd = "module purge; module load gcc rstats/4.5.1 || module load r/4.5.1 || module load r; cd ${RemoteBase}/code_r && R CMD SHLIB selection_utils.cpp -o selection_utils.so && R CMD SHLIB proj_simplex.cpp -o proj_simplex.so"
ssh "${Username}@${Hostname}" $compileCmd
Write-Host "Compilation complete." -ForegroundColor Green

# 7. Submit array jobs
Write-Host "Submitting array jobs..." -ForegroundColor Cyan
$submittedJobs = @()

foreach ($sep in $Separations) {
    foreach ($p in $Dimensions) {
        Write-Host "  Submitting: sep=$sep, p=$p, noise=$Noise, methods=$Methods ..."

        $logPattern = "logs/sim_id%a_sep${sep}_p${p}_${Noise}"
        $sbatchCmd = "cd ${RemoteBase}/simulations/production/${SimName} && sbatch --output=${logPattern}.out --error=${logPattern}.err --export=ALL,SEP=$sep,P=$p,NOISE=$Noise,METHODS=$Methods submit.sh"

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
