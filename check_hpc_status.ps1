<#
.SYNOPSIS
    Checks the status of SLURM jobs on the HPC cluster.

.DESCRIPTION
    Runs 'squeue -u <username>' on the remote server via SSH.

.PARAMETER Username
    The SSH username. default: jongminm

.PARAMETER Hostname
    The HPC hostname. default: discovery.usc.edu

.EXAMPLE
    .\check_hpc_status.ps1
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu"
)

$ErrorActionPreference = "Stop"

Write-Host "Checking job queue for $Username on $Hostname..." -ForegroundColor Cyan

# Run squeue command
ssh "${Username}@${Hostname}" "squeue -u ${Username}"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nStatus check complete." -ForegroundColor Green
}
else {
    Write-Error "Failed to check status."
}
