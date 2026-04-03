param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$ErrorActionPreference = "Stop"

$SimName = "competitors_knowncov"
$LocalDir = "d:\GitHub\sparse_kmeans\simulations\$SimName"
$RemoteDir = "${RemoteBase}/simulations/${SimName}"

Write-Host "Retrieving simulation results and logs from HPC..." -ForegroundColor Cyan

# Prepare local repository struct internally to absorb outputs natively
# This keeps the results_raw hierarchy (noise/p/sim_job...)
New-Item -ItemType Directory -Force -Path "${LocalDir}\results_raw" -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Force -Path "${LocalDir}\logs" -ErrorAction SilentlyContinue | Out-Null

Write-Host "Syncing results_raw..."
# Using scp -r to pull the entire results_raw directory
scp -r "${Username}@${Hostname}:${RemoteDir}/results_raw/*" "${LocalDir}/results_raw/"

Write-Host "Syncing logs..."
scp -r "${Username}@${Hostname}:${RemoteDir}/logs/*" "${LocalDir}/logs/"

Write-Host "Sync process complete. You can now execute aggregate_hpc.R locally." -ForegroundColor Green
