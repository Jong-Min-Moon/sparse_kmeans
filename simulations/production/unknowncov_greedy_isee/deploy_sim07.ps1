<#
.SYNOPSIS
    Deploys Simulation 07 (SAM FDR) to HPC and submits jobs for multiple FDR targets.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [double[]]$FdrTargets = @(0.4)
)

$ErrorActionPreference = "Stop"

# Define Local Paths
$SimName = "unknowncov_greedy_isee"
$SimDir = "d:\GitHub\sparse_kmeans\simulations\production\$SimName"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# --- 1. SSH Key Authentication Setup ---
Write-Host "Checking SSH Key Authentication..." -ForegroundColor Cyan

$sshKeyPathRsa = Join-Path $env:USERPROFILE ".ssh\id_rsa"
$sshKeyPathEd25519 = Join-Path $env:USERPROFILE ".ssh\id_ed25519"
$keyExists = (Test-Path $sshKeyPathRsa) -or (Test-Path $sshKeyPathEd25519)

if (-not $keyExists) {
    Write-Host "No SSH key found. Generating a new ed25519 key..."
    ssh-keygen -t ed25519 -f (Join-Path $env:USERPROFILE ".ssh\id_ed25519") -N '""' -q
    Write-Host "Key generated."
}

# Check if we can connect without password
$sshCheck = ""
try {
    $sshCheck = ssh -o BatchMode=yes -o ConnectTimeout=5 "$Username@$Hostname" "echo OK" 2>&1
}
catch {
    $sshCheck = "Failed"
}
if ($sshCheck -notlike "*OK*") {
    Write-Host "Setting up passwordless SSH login..." -ForegroundColor Yellow
    $pubKeyPath = if (Test-Path "$sshKeyPathEd25519.pub") { "$sshKeyPathEd25519.pub" } else { "$sshKeyPathRsa.pub" }
    $pubKey = Get-Content $pubKeyPath -Raw
    ssh "$Username@$Hostname" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$pubKey' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
}
else {
    Write-Host "Passwordless SSH login is already configured." -ForegroundColor Green
}

# --- 2. Deployment ---
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "$Username@$Hostname" "mkdir -p $RemoteBase/simulations/production/$SimName && mkdir -p $RemoteBase/code_r"

Write-Host "Transferring files..." -ForegroundColor Cyan
scp "$SimDir\driver.R" "$SimDir\submit.sh" "${Username}@${Hostname}:${RemoteBase}/simulations/production/$SimName/" | Out-Null
scp "$CodeDir\*.R" "${Username}@${Hostname}:${RemoteBase}/code_r/" | Out-Null
scp "$CodeDir\*.cpp" "${Username}@${Hostname}:${RemoteBase}/code_r/" | Out-Null

Write-Host "Preparing HPC environment..." -ForegroundColor Cyan
$prepCmd = "cd $RemoteBase/simulations/production/$SimName && rm -rf logs results results_aggregated && mkdir -p logs results && dos2unix *.sh *.R 2>/dev/null && chmod +x *.sh"
ssh "$Username@$Hostname" $prepCmd

# --- 2b. Compile Rcpp C++ backend on HPC ---
Write-Host "Compiling C++ source files on the HPC..." -ForegroundColor Cyan
$compileCmd = "module purge; module load gcc rstats/4.5.1 || module load r/4.5.1 || module load r; cd $RemoteBase/code_r && R CMD SHLIB selection_utils.cpp -o selection_utils.so"
ssh "$Username@$Hostname" $compileCmd
Write-Host "Compilation complete." -ForegroundColor Green

# --- 3. Submit Jobs ---
Write-Host "Submitting jobs..." -ForegroundColor Cyan
$submittedJobs = @()

foreach ($fdr in $FdrTargets) {
    Write-Host "Submitting simulation for FDR Target: $fdr"
    $fdrStr = [string]$fdr -replace '\.', 'p'
    $sbatchCmd = "cd $RemoteBase/simulations/production/$SimName && sbatch --output=logs/sim_id%a_fdr$fdrStr.out --error=logs/sim_id%a_fdr$fdrStr.err --export=ALL,FDR_TARGET=$fdr submit.sh"
    
    $output = ssh "$Username@$Hostname" $sbatchCmd 2>&1
    
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
