
<#
.SYNOPSIS
    Retrieves Simulation 08 (Oracle ISEE) results from the HPC cluster.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$ErrorActionPreference = "Stop"

$LocalDir = "d:\GitHub\sparse_kmeans\simulations\sim08_true_label_ISEE"
$RemoteResultsDir = "${RemoteBase}/simulations/sim08_true_label_ISEE/results"
$LocalOutputDir = "${LocalDir}\output"

Write-Host "Retrieving results from ${Username}@${Hostname}:${RemoteResultsDir} ..." -ForegroundColor Cyan
Write-Host "Local destination: ${LocalOutputDir}\" -ForegroundColor Cyan

if (!(Test-Path $LocalOutputDir)) {
    New-Item -ItemType Directory -Path $LocalOutputDir | Out-Null
}

Write-Host "Copying results directory..." -ForegroundColor Gray
scp -r "${Username}@${Hostname}:${RemoteResultsDir}" "${LocalDir}/"
$exitCode = $LASTEXITCODE

if (Test-Path "${LocalDir}\results") {
    if (Test-Path $LocalOutputDir) { Remove-Item -Recurse -Force $LocalOutputDir }
    Rename-Item "${LocalDir}\results" "output"
}

if ($exitCode -eq 0) {
    Write-Host "`nResults retrieved successfully!" -ForegroundColor Green
    Write-Host "Results saved to: ${LocalOutputDir}" -ForegroundColor Green
}
else {
    Write-Error "Failed to retrieve results."
}
