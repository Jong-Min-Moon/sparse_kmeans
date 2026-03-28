param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$LocalDir = "d:\GitHub\sparse_kmeans\simulations\sim_22_thompson_identity_laplace"
$RemoteDir = "${RemoteBase}/simulations/sim_22_thompson_identity_laplace/results_raw"

Write-Host "Retrieving scheduled simulation results back from HPC..." -ForegroundColor Cyan

# Prepare local repository struct internally to absorb outputs natively
New-Item -ItemType Directory -Force -Path "${LocalDir}\results_raw" -ErrorAction SilentlyContinue | Out-Null

scp -r "${Username}@${Hostname}:${RemoteDir}/*" "${LocalDir}\results_raw\"

Write-Host "Sync process complete. You can now execute aggregate_sim_laplace.R globally." -ForegroundColor Green
