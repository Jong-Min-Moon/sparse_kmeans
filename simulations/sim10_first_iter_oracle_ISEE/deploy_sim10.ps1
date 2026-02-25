
<#
.SYNOPSIS
    Deploys Simulation 10 (First-Iter Oracle ISEE, Thompson v3.1) to HPC.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$ErrorActionPreference = "Stop"

$SimDir = "d:\GitHub\sparse_kmeans\simulations\sim10_first_iter_oracle_ISEE"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Transfer files (includes the new first-iter oracle Thompson algorithm)
Write-Host "Transferring files..." -ForegroundColor Cyan

$FilesToTransfer = @("${SimDir}\run_sim_v3_1.R")
$FilesToTransfer += Get-Item "${CodeDir}\*.R"
$FilesToTransfer += Get-Item "${CodeDir}\*.cpp"

ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/staging"
scp $FilesToTransfer "${Username}@${Hostname}:${RemoteBase}/staging/"

# 2. Remote: organize, compile, generate slurm
Write-Host "Organizing files, compiling, and generating slurm..." -ForegroundColor Cyan

$remoteCmd = @'
set -e
RBASE=$(eval echo {{REMOTE_BASE}})
SIMDIR=$RBASE/simulations/sim10_first_iter_oracle_ISEE
CODEDIR=$RBASE/code_r

mkdir -p $SIMDIR $CODEDIR
mv $RBASE/staging/run_sim_v3_1.R $SIMDIR/
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

# Write slurm via printf
cd $SIMDIR
rm -rf logs results
mkdir -p logs results
sed -i 's/\r$//' run_sim_v3_1.R

printf '#!/bin/bash\n#SBATCH --job-name=sim10_first_oracle\n#SBATCH --partition=main\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=10\n#SBATCH --mem=6G\n#SBATCH --time=23:00:00\n#SBATCH --output=logs/sim_id%%a.out\n#SBATCH --error=logs/sim_id%%a.err\n#SBATCH --array=1-10\n\nmodule load rstats/4.5.1\nmkdir -p logs results\nRscript run_sim_v3_1.R --job_id $SLURM_ARRAY_TASK_ID --p_val 0.001 --perms 10000 --C 0.3\n' > submit_sim10.slurm

chmod +x submit_sim10.slurm
echo "SETUP COMPLETE"
'@

$sanitizedCmd = ($remoteCmd -replace '{{REMOTE_BASE}}', $RemoteBase) -replace "`r", ""
$sanitizedCmd | ssh "${Username}@${Hostname}" "bash -l"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Setup/compilation failed."
    exit 1
}

# 3. Submit via separate SSH call
Write-Host "Submitting job..." -ForegroundColor Cyan
$resolvedBase = ssh "${Username}@${Hostname}" "eval echo ${RemoteBase}"
ssh "${Username}@${Hostname}" "cd ${resolvedBase}/simulations/sim10_first_iter_oracle_ISEE && sbatch submit_sim10.slurm"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Job submitted successfully!" -ForegroundColor Green
}
else {
    Write-Error "Job submission failed."
}
