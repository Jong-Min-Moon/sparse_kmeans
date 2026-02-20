<#
.SYNOPSIS
    Retrieves Simulation 05 results from the HPC cluster.

.DESCRIPTION
    Uses 'scp' to copy all results from the remote server to the local 'simulations/sim05_unknown_cov_deterministic_n200' directory.

.PARAMETER Username
    The SSH username. default: jongminm

.PARAMETER Hostname
    The HPC hostname. default: discovery.usc.edu

.PARAMETER RemoteBase
    The base directory on the remote server where simulation data is stored.
    Results are expected in <RemoteBase>/simulations/sim05_unknown_cov_deterministic_n200/results/.
    default: ~/sparse_kmeans_project

.PARAMETER LocalDir
    The local directory to download results to.
    default: d:\GitHub\sparse_kmeans\simulations\sim05_unknown_cov_deterministic_n200

.EXAMPLE
    .\retrieve_sim05.ps1
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [string]$LocalDir = "d:\GitHub\sparse_kmeans\simulations\sim05_unknown_cov_deterministic_n200"
)

$ErrorActionPreference = "Stop"

$RemoteSimDir = "${RemoteBase}/simulations/sim05_unknown_cov_deterministic_n200"
$RemoteResultsDir = "${RemoteSimDir}/results"

Write-Host "Retrieving results from ${Username}@${Hostname}:${RemoteResultsDir} ..." -ForegroundColor Cyan
Write-Host "Local destination: $LocalDir\output\" -ForegroundColor Cyan

# Create local output directory if it doesn't exist
$localOutputDir = Join-Path $LocalDir "output"
if (-not (Test-Path $localOutputDir)) {
    New-Item -ItemType Directory -Path $localOutputDir | Out-Null
}

# Copy all results
Write-Host "Copying results directory..." -ForegroundColor Gray
# Using proper variable expansion for scp
scp -r "${Username}@${Hostname}:${RemoteResultsDir}" "$LocalDir/"

# Optionally rename to output for consistency
if (Test-Path (Join-Path $LocalDir "results")) {
    if (Test-Path $localOutputDir) {
        Remove-Item -Recurse -Force $localOutputDir
    }
    Rename-Item -Path (Join-Path $LocalDir "results") -NewName "output"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nResults retrieved successfully!" -ForegroundColor Green
    Write-Host "Results saved to: $localOutputDir" -ForegroundColor Green
}
else {
    Write-Error "Failed to retrieve results. Check your connection or if results exist on the server."
}
