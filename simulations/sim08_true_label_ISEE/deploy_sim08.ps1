
<#
.SYNOPSIS
    Deploys Simulation 08 (Oracle ISEE) to HPC.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$ErrorActionPreference = "Stop"

$SimDir = "d:\GitHub\sparse_kmeans\simulations\sim08_true_label_ISEE"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Transfer R files (including the oracle ISEE algorithm)
Write-Host "Transferring files..." -ForegroundColor Cyan

$FilesToTransfer = @("${SimDir}\driver.R")
$FilesToTransfer += Get-Item "${CodeDir}\*.R"
$FilesToTransfer += Get-Item "${CodeDir}\*.cpp"

ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/staging"
scp $FilesToTransfer "${Username}@${Hostname}:${RemoteBase}/staging/"

# 2. Remote: organize files, compile C++, generate slurm, and submit
Write-Host "Organizing files, compiling, and generating slurm..." -ForegroundColor Cyan

$remoteCmd = @'
set -e
RBASE=$(eval echo {{REMOTE_BASE}})
SIMDIR=$RBASE/simulations/sim08_true_label_ISEE
CODEDIR=$RBASE/code_r

mkdir -p $SIMDIR $CODEDIR
mv $RBASE/staging/driver.R $SIMDIR/
mv $RBASE/staging/*.R $CODEDIR/
mv $RBASE/staging/*.cpp $CODEDIR/
rmdir $RBASE/staging

# Compile C++ backend
cd $CODEDIR
rm -f selection_utils.o selection_utils.so selection_utils.dll Makevars
module load rstats/4.5.1
Rscript -e 'writeLines(c(paste0("PKG_CXXFLAGS = -fopenmp -I", system.file("include", package="Rcpp")), "PKG_LIBS = -fopenmp"), "Makevars")'
R CMD SHLIB selection_utils.cpp
if [ ! -f selection_utils.so ]; then echo "Error: selection_utils.so not generated"; exit 1; fi
rm -f Makevars

# Write slurm file via printf
cd $SIMDIR
rm -rf logs results
mkdir -p logs results
sed -i 's/\r$//' driver.R

printf '#!/bin/bash\n#SBATCH --job-name=sim08_oracle_isee\n#SBATCH --partition=main\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=10\n#SBATCH --mem=8G\n#SBATCH --time=23:00:00\n#SBATCH --output=logs/sim_id%%a.out\n#SBATCH --error=logs/sim_id%%a.err\n#SBATCH --array=1-100\n\nmodule load rstats/4.5.1\nmkdir -p logs results\nRscript driver.R --job_id $SLURM_ARRAY_TASK_ID --fdr 0.4 --perms 20\n' > submit.sh

chmod +x submit.sh
echo "SETUP COMPLETE"
'@

$sanitizedCmd = ($remoteCmd -replace '{{REMOTE_BASE}}', $RemoteBase) -replace "`r", ""
$sanitizedCmd | ssh "${Username}@${Hostname}" "bash -l"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Setup/compilation failed."
    exit 1
}

# 3. Submit via separate SSH call (sbatch needs clean stdin)
Write-Host "Submitting job..." -ForegroundColor Cyan
$resolvedBase = ssh "${Username}@${Hostname}" "eval echo ${RemoteBase}"
ssh "${Username}@${Hostname}" "cd ${resolvedBase}/simulations/sim08_true_label_ISEE && sbatch submit.sh"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Job submitted successfully!" -ForegroundColor Green
}
else {
    Write-Error "Job submission failed."
}
