<#
.SYNOPSIS
    Retrieves results from Greedy ISEE n200-Chain simulation.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$SimName = "unknowncov_n200_chain_greedy_isee"
$SimDir = "d:\GitHub\sparse_kmeans\simulations\production\$SimName"

Write-Host "Retrieving results from HPC..." -ForegroundColor Cyan

# Use rsync if possible, otherwise scp
# scp -r "${Username}@${Hostname}:${RemoteBase}/simulations/production/${SimName}/results_raw" "${SimDir}\"

$rsyncCmd = "rsync -avz --progress ${Username}@${Hostname}:${RemoteBase}/simulations/production/${SimName}/results_raw/ ${SimDir}/results_raw/"
Write-Host "Running: $rsyncCmd"
Invoke-Expression $rsyncCmd

Write-Host "Retrieval complete." -ForegroundColor Green
