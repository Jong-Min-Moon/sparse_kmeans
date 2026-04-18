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

Write-Host "Streaming packaged tarball natively over SSH to localhost..." -ForegroundColor Cyan
cmd.exe /c "ssh ${Username}@${Hostname} ""cd ${RemoteDir} && tar -cz results_raw logs"" > ""${LocalDir}\retrieve_package.tar.gz"""

Write-Host "Extracting archive natively..." -ForegroundColor Cyan
tar -xzf "${LocalDir}\retrieve_package.tar.gz" -C "${LocalDir}"

Write-Host "Cleaning up tarball..." -ForegroundColor Cyan
Remove-Item -Force "${LocalDir}\retrieve_package.tar.gz"

Write-Host "Retrieval complete. You can now execute aggregation scripts locally." -ForegroundColor Green
