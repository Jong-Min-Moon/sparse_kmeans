param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$LocalDir = "d:\GitHub\sparse_kmeans\simulations\sim14_thompson"
$RemoteDir = "${RemoteBase}/simulations/sim14_thompson/results"

Write-Host "Retrieving simulation results..." -ForegroundColor Cyan
mkdir.exe -p "${LocalDir}\output" -ErrorAction SilentlyContinue

scp -r "${Username}@${Hostname}:${RemoteDir}/*.rds" "${LocalDir}\output\"

Write-Host "Done." -ForegroundColor Green
