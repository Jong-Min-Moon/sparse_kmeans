<#
.SYNOPSIS
    Retrieves Simulation 01 results from the HPC cluster.
    Mirrors retrieve_sim_laplace.ps1 conventions.
#>

param(
    [string]$Username   = "jongminm",
    [string]$Hostname   = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$LocalDir  = "d:\GitHub\sparse_kmeans\simulations\production\sim01_greedy_varying_p"
$RemoteDir = "${RemoteBase}/simulations/production/sim01_greedy_varying_p/results_raw"

Write-Host "Retrieving sim01 results from HPC..." -ForegroundColor Cyan

# Ensure local results_raw directory exists
New-Item -ItemType Directory -Force -Path "${LocalDir}\results_raw" -ErrorAction SilentlyContinue | Out-Null

scp -r "${Username}@${Hostname}:${RemoteDir}/*" "${LocalDir}\results_raw\"

Write-Host "Sync complete. Run aggregate_sim01.R to compile results." -ForegroundColor Green

