param(
    [string]$ServerUsername = "jongminm",
    [string]$ServerAddress = "hpc.usc.edu"
)

$RemotePath = "/home1/$ServerUsername/sparse_kmeans/simulations/sim_15_witten_arias/results/"
$LocalPath = "results/"

Write-Host "Creating local results directory..."
New-Item -ItemType Directory -Force -Path $LocalPath

Write-Host "Retrieving results from HPC..."
scp -r "${ServerUsername}@${ServerAddress}:${RemotePath}*" $LocalPath

Write-Host "Done!"
