<#
.SYNOPSIS
    Deploys Simulation 07 (SAM FDR) to HPC and submits jobs for multiple FDR targets.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [double[]]$FdrTargets = @(0.4, 0.5, 0.6)
)

$ErrorActionPreference = "Stop"

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
    # It is expected to fail or throw an exception if keys aren't set
    $sshCheck = "Failed"
}
if ($sshCheck -notlike "*OK*") {
    Write-Host "Setting up passwordless SSH login... (You will be prompted for your password ONE last time)" -ForegroundColor Yellow
    # Read public key
    $pubKeyPath = if (Test-Path "$sshKeyPathEd25519.pub") { "$sshKeyPathEd25519.pub" } else { "$sshKeyPathRsa.pub" }
    $pubKey = Get-Content $pubKeyPath -Raw
    
    # Use standard ssh to append key to authorized_keys (prompts for password)
    ssh "$Username@$Hostname" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$pubKey' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    
    # Verify
    $sshCheck2 = ssh -o BatchMode=yes -o ConnectTimeout=5 "$Username@$Hostname" "echo OK" 2>&1
    if ($sshCheck2 -notlike "*OK*") {
        Write-Error "Failed to set up SSH key authentication. Please check your credentials and try again."
        exit 1
    }
    Write-Host "SSH key setup successful! Passwordless login enabled." -ForegroundColor Green
}
else {
    Write-Host "Passwordless SSH login is already configured." -ForegroundColor Green
}

# --- 2. Deployment ---
$SimDir = "d:\GitHub\sparse_kmeans\simulations\sim07_SAM_sep3"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "$Username@$Hostname" "mkdir -p $RemoteBase/simulations/sim07_SAM_sep3 && mkdir -p $RemoteBase/code_r && mkdir -p $RemoteBase/simulations/sim07_SAM_sep3/logs"

Write-Host "Transferring files..." -ForegroundColor Cyan
scp "$SimDir\driver.R" "$SimDir\submit.sh" "${Username}@${Hostname}:${RemoteBase}/simulations/sim07_SAM_sep3/" | Out-Null
scp "$CodeDir\*.R" "${Username}@${Hostname}:${RemoteBase}/code_r/" | Out-Null

Write-Host "Preparing HPC environment..." -ForegroundColor Cyan
ssh "$Username@$Hostname" "cd $RemoteBase/simulations/sim07_SAM_sep3 && dos2unix *.sh *.R 2>/dev/null && chmod +x *.sh"

# --- 3. Submit Jobs for Multiple FDR Targets ---
Write-Host "Submitting jobs..." -ForegroundColor Cyan
$submittedJobs = @()

foreach ($fdr in $FdrTargets) {
    Write-Host "Submitting simulation for FDR Target: $fdr"
    
    # Format fdr string for log filenames (0.4 -> 0p4)
    $fdrStr = [string]$fdr -replace '\.', 'p'
    
    # Construct SBATCH command bypassing the static headers for dynamic logs and exporting the env var
    $sbatchCmd = "cd $RemoteBase/simulations/sim07_SAM_sep3 && sbatch --output=logs/sim_id%a_fdr$fdrStr.out --error=logs/sim_id%a_fdr$fdrStr.err --export=ALL,FDR_TARGET=$fdr submit.sh"
    
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
Write-Host "SSH Authentication: Key-based (Passwordless)"
Write-Host "===================================================="
