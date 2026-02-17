<#
.SYNOPSIS
    Retrieves Simulation 01 results from the HPC cluster.

.DESCRIPTION
    Uses 'scp' to copy all 'output_p*' folders from the remote server to the local 'simulations/sim01_greedy_varying_p' directory.

.PARAMETER Username
    The SSH username. default: jongminm

.PARAMETER Hostname
    The HPC hostname. default: discovery.usc.edu

.PARAMETER RemoteBase
    The base directory on the remote server where simulation data is stored.
    Results are expected in <RemoteBase>/simulations/sim01_greedy_varying_p/output_p*.
    default: ~/sparse_kmeans_project

.PARAMETER LocalDir
    The local directory to download results to.
    default: d:\GitHub\sparse_kmeans\simulations\sim01_greedy_varying_p

.EXAMPLE
    .\retrieve_sim01.ps1
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project",
    [string]$LocalDir = "d:\GitHub\sparse_kmeans\simulations\sim01_greedy_varying_p"
)

$ErrorActionPreference = "Stop"

$RemoteSimDir = "${RemoteBase}/simulations/sim01_greedy_varying_p"

Write-Host "Retrieving all 'output_p*' results from ${Username}@${Hostname}:${RemoteSimDir} ..." -ForegroundColor Cyan
Write-Host "Local destination: $LocalDir" -ForegroundColor Cyan

# Use SCP to copy all output_p* directories
# Note: Wildcards in SCP can be tricky; it's often safer to copy them one by one or the whole subdir if it's clean.
# Since we only want the output_p* folders, we'll use a remote command to list them and loop, 
# or just try to scp with the pattern (which works in many shells).

# We will try a simple recursive scp of the remote patterns.
# SCP doesn't always handle remote wildcards natively unless quoted or handled by the shell.
# A more robust way is to use a single SCP command for each directory.

$remoteCmd = "ls -d ${RemoteSimDir}/output_p*"
$outputPaths = ssh "${Username}@${Hostname}" $remoteCmd

foreach ($remotePath in $outputPaths) {
    if ($remotePath -match "output_p") {
        Write-Host "Copying $remotePath ..." -ForegroundColor Gray
        scp -r "${Username}@${Hostname}:${remotePath}" "$LocalDir/"
    }
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nResults retrieved successfully!" -ForegroundColor Green
}
else {
    Write-Error "Failed to retrieve results. Check your connection or if results exist on the server."
}
