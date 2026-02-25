
<#
.SYNOPSIS
    Retrieves Simulation 07 results from the HPC cluster.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$ErrorActionPreference = "Stop"

$LocalDir = "d:\GitHub\sparse_kmeans\simulations\sim11_greedy_warmstart_TVS"
$RemoteSimDir = "${RemoteBase}/simulations/sim11_greedy_warmstart_TVS"
$RemoteResultsDir = "${RemoteSimDir}/results"
$LocalOutputDir = "${LocalDir}\output"

Write-Host "Retrieving results from ${Username}@${Hostname}:${RemoteResultsDir} ..." -ForegroundColor Cyan
Write-Host "Local destination: ${LocalOutputDir}\" -ForegroundColor Cyan

# Create output directory
if (!(Test-Path $LocalOutputDir)) {
    New-Item -ItemType Directory -Path $LocalOutputDir | Out-Null
}

# Copy results from HPC
Write-Host "Copying results directory..." -ForegroundColor Gray
scp -r "${Username}@${Hostname}:${RemoteResultsDir}" "${LocalDir}/"
$exitCode = $LASTEXITCODE

# Rename results -> output
if (Test-Path "${LocalDir}\results") {
    if (Test-Path $LocalOutputDir) {
        Remove-Item -Recurse -Force $LocalOutputDir
    }
    Rename-Item "${LocalDir}\results" "output"
}

if ($exitCode -eq 0) {
    Write-Host "`nResults retrieved successfully!" -ForegroundColor Green
    Write-Host "Results saved to: ${LocalOutputDir}" -ForegroundColor Green
}
else {
    Write-Error "Failed to retrieve results."
}
