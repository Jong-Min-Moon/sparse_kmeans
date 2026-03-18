param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$LocalDir = "d:\GitHub\sparse_kmeans\simulations\sim_cluster_greedy_varying_p"
$RemoteDir = "${RemoteBase}/simulations/sim_cluster_greedy_varying_p/results_raw"

Write-Host "Retrieving scheduled simulation results back from HPC..." -ForegroundColor Cyan

# Prepare local repository struct internally to absorb outputs natively
mkdir.exe -p "${LocalDir}\results_raw" -ErrorAction SilentlyContinue

scp -r "${Username}@${Hostname}:${RemoteDir}/*" "${LocalDir}\results_raw\"

Write-Host "Sync process complete. You can now execute aggregate_greedy.R globally." -ForegroundColor Green
