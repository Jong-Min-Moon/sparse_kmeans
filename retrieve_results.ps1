<#
.SYNOPSIS
    Retrieves simulation results from the HPC cluster.

.DESCRIPTION
    Uses 'scp' to copy the 'results' folder from the remote server to the local machine.
    
.PARAMETER Username
    The SSH username. default: jongminm

.PARAMETER Hostname
    The HPC hostname. default: discovery.usc.edu

.PARAMETER RemoteDir
    The base directory on the remote server where simulation data is stored.
    Results are expected in <RemoteDir>/results.
    default: ~/sparse_kmeans_sim

.PARAMETER LocalDir
    The local directory to download results to.
    default: ./hpc_results

.EXAMPLE
    .\retrieve_results.ps1
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteDir = "~/sparse_kmeans_sim",
    [string]$LocalDir = "./hpc_results"
)

$ErrorActionPreference = "Stop"

Write-Host "Retrieving results from ${Username}@${Hostname}:${RemoteDir}/results ..." -ForegroundColor Cyan
Write-Host "Local destination: $LocalDir" -ForegroundColor Cyan

# Ensure local directory exists
if (-not (Test-Path $LocalDir)) {
    New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
}

# Run SCP command
scp -r "${Username}@${Hostname}:${RemoteDir}/results/" "$LocalDir"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nResults retrieved successfully!" -ForegroundColor Green
    Write-Host "Files are located in: $(Resolve-Path $LocalDir)" -ForegroundColor Green
}
else {
    Write-Error "Failed to retrieve results. Check your connection or if results exist on the server."
}
