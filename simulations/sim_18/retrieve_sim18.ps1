param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$LocalDir = "d:\GitHub\sparse_kmeans\simulations\sim_18"
$RemoteDir = "${RemoteBase}/simulations/sim_18/results_raw"

Write-Host "Retrieving simulation results..." -ForegroundColor Cyan
mkdir.exe -p "${LocalDir}\results_raw" -ErrorAction SilentlyContinue

scp -r "${Username}@${Hostname}:${RemoteDir}/*.rds" "${LocalDir}\results_raw\"

Write-Host "Done." -ForegroundColor Green
