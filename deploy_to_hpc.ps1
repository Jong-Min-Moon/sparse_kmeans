<#
.SYNOPSIS
    Automates the deployment of R simulation code to an HPC cluster and submits the job.

.DESCRIPTION
    This script uses SCP to copy the 'code_r' directory to a specified remote directory
    on an HPC cluster and then uses SSH to submit the 'submit_job.sh' script via 'sbatch'.

.PARAMETER Username
    The SSH username for the HPC cluster.

.PARAMETER Hostname
    The hostname or IP address of the HPC cluster.

.PARAMETER RemoteDir
    The destination directory on the remote server. Defaults to "~/sparse_kmeans_sim".

.EXAMPLE
    .\deploy_to_hpc.ps1 -Username "jdoe" -Hostname "hpc.university.edu"
#>

param(
    [Parameter(Mandatory = $false)]
    [string]$Username = "jongminm",

    [Parameter(Mandatory = $false)]
    [string]$Hostname = "discovery.usc.edu",

    [string]$RemoteDir = "~/sparse_kmeans_sim"
)

$ErrorActionPreference = "Stop"

# 1. Create directory first (ignore error if exists)
Write-Host "Creating remote directory (if needed)..." -ForegroundColor Cyan
try {
    ssh "${Username}@${Hostname}" "mkdir -p ${RemoteDir}"
}
catch {
    Write-Warning "Directory creation might have failed or already exists. Continuing..."
}

# 2. Transfer code using SCP
Write-Host "Transferring code to ${Username}@${Hostname}:${RemoteDir}/code_r ..." -ForegroundColor Cyan
# Ensure trailing slash on source to copy CONTENTS of code_r into destination
scp -r "d:\GitHub\sparse_kmeans\code_r" "${Username}@${Hostname}:${RemoteDir}/"

if ($LASTEXITCODE -ne 0) {
    Write-Error "SCP transfer failed."
}

# 2. Fix line endings (Windows -> Unix)
# Ideally done locally, but can be done remotely.
# We'll assume the user has dos2unix available on the server or git bash handles it.
# Or use sed remotely.

# 3. Submit Job
Write-Host "Submitting job on cluster..." -ForegroundColor Cyan
$submitCmd = "cd $RemoteDir/code_r && dos2unix submit_job.sh && sbatch submit_job.sh"

ssh "$Username@$Hostname" $submitCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Job submitted successfully!" -ForegroundColor Green
}
else {
    Write-Error "Job submission failed."
}
