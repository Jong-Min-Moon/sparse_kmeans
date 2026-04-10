<#
.SYNOPSIS
    Retrieves unknowncov_greedy_naive results from the HPC cluster.
    Mirrors retrieve_unknowncov.ps1 conventions.
#>

param(
    [string]$Username   = "jongminm",
    [string]$Hostname   = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$ErrorActionPreference = "Stop"

$SimName   = "unknowncov_greedy_naive"
$LocalDir  = "d:\GitHub\sparse_kmeans\simulations\production\$SimName"
$RemoteDir = "${RemoteBase}/simulations/production/${SimName}"

Write-Host "Retrieving simulation results and logs from HPC..." -ForegroundColor Cyan

# Ensure local directory structure exists
New-Item -ItemType Directory -Force -Path "${LocalDir}\results_raw" -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Force -Path "${LocalDir}\logs"        -ErrorAction SilentlyContinue | Out-Null

Write-Host "Syncing results_raw..."
scp -r "${Username}@${Hostname}:${RemoteDir}/results_raw/*" "${LocalDir}/results_raw/"

Write-Host "Syncing logs..."
scp -r "${Username}@${Hostname}:${RemoteDir}/logs/*" "${LocalDir}/logs/"

Write-Host "Sync complete. Run aggregate.R to compile results." -ForegroundColor Green
