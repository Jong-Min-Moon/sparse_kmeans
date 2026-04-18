param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$ErrorActionPreference = "Stop"

$SimName = "unknowncov_n500_chain_competitors_modern"
$LocalDir = "d:\GitHub\sparse_kmeans\simulations\production\$SimName"
$RemoteDir = "${RemoteBase}/simulations/production/${SimName}"

Write-Host "Retrieving simulation results and logs from HPC..." -ForegroundColor Cyan

# Prepare local repository struct internally to absorb outputs natively
# This keeps the results_raw hierarchy (noise/p/sim_job...)
New-Item -ItemType Directory -Force -Path "${LocalDir}\results_raw" -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Force -Path "${LocalDir}\logs" -ErrorAction SilentlyContinue | Out-Null
Write-Host "Streaming packaged tarball natively over SSH to localhost..." -ForegroundColor Cyan
cmd.exe /c "ssh ${Username}@${Hostname} ""cd ${RemoteDir} && tar -cz results_raw logs"" > ""${LocalDir}\retrieve_package.tar.gz"""

Write-Host "Extracting archive natively..." -ForegroundColor Cyan
tar -xzf "${LocalDir}\retrieve_package.tar.gz" -C "${LocalDir}"

Write-Host "Cleaning up tarball..." -ForegroundColor Cyan
Remove-Item -Force "${LocalDir}\retrieve_package.tar.gz"

Write-Host "Sync process complete. You can now execute aggregate_hpc.R locally." -ForegroundColor Green
