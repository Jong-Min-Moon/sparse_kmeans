<#
.SYNOPSIS
    Retrieves results from Greedy ISEE n200-Chain simulation.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$ErrorActionPreference = "Stop"

$SimName = Split-Path -Leaf $PSScriptRoot
$LocalDir = $PSScriptRoot
$RemoteDir = "${RemoteBase}/simulations/production/${SimName}"

Write-Host "Retrieving simulation results and logs from HPC..." -ForegroundColor Cyan

# Ensure local directory structure exists
New-Item -ItemType Directory -Force -Path "${LocalDir}\results_raw" -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Force -Path "${LocalDir}\logs" -ErrorAction SilentlyContinue | Out-Null

Write-Host "Syncing results_raw..."
# Using scp -r to pull the entire results_raw directory
scp -r "${Username}@${Hostname}:${RemoteDir}/results_raw/*" "${LocalDir}/results_raw/"

Write-Host "Syncing logs..."
scp -r "${Username}@${Hostname}:${RemoteDir}/logs/*" "${LocalDir}/logs/"

Write-Host "Retrieval complete. You can now execute aggregation scripts locally." -ForegroundColor Green
