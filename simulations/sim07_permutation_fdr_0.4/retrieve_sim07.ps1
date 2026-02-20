<#
.SYNOPSIS
    Retrieves Simulation 07 results from the HPC cluster.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [string]$LocalDir = "d:\GitHub\sparse_kmeans\simulations\sim07_permutation_fdr_0.4"
)

$ErrorActionPreference = "Stop"

$RemoteSimDir = "${RemoteBase}/simulations/sim07_permutation_fdr_0.4"
$RemoteResultsDir = "${RemoteSimDir}/results"

Write-Host "Retrieving results from ${Username}@${Hostname}:${RemoteResultsDir} ..." -ForegroundColor Cyan
Write-Host "Local destination: $LocalDir\output\" -ForegroundColor Cyan

# Create local output directory
$localOutputDir = Join-Path $LocalDir "output"
if (-not (Test-Path $localOutputDir)) {
    New-Item -ItemType Directory -Path $localOutputDir | Out-Null
}

# Copy all results
Write-Host "Copying results directory..." -ForegroundColor Gray
scp -r "${Username}@${Hostname}:${RemoteResultsDir}" "$LocalDir/"

# Rename to output
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
    Write-Error "Failed to retrieve results."
}
