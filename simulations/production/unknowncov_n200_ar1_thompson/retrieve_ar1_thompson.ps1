param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$LocalDir = "d:\GitHub\sparse_kmeans\simulations\production\unknowncov_n200_ar1_thompson"
$RemoteDir = "${RemoteBase}/simulations/production/unknowncov_n200_ar1_thompson"

Write-Host "Retrieving scheduled simulation results back from HPC..." -ForegroundColor Cyan

# Ensure local directory structure exists
New-Item -ItemType Directory -Force -Path "${LocalDir}\results_raw" -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Force -Path "${LocalDir}\logs" -ErrorAction SilentlyContinue | Out-Null

Write-Host "Syncing results_raw..."
scp -r "${Username}@${Hostname}:${RemoteDir}/results_raw/*" "${LocalDir}/results_raw/"

Write-Host "Syncing logs..."
scp -r "${Username}@${Hostname}:${RemoteDir}/logs/*" "${LocalDir}/logs/"

Write-Host "Sync process complete. You can now execute aggregate_ar1_thompson.R globally." -ForegroundColor Green
