
<#
.SYNOPSIS
    Deploys Simulation 02 (Unknown Covariance Greedy, 100 reps) to HPC.

.DESCRIPTION
    1. Copies the 'sim02_unknown_cov_greedy' folder to ~/sparse_kmeans_project/simulations/sim02_unknown_cov_greedy
    2. Copies the 'code_r' folder to ~/sparse_kmeans_project/code_r (to ensure dependencies are up to date)
    3. Runs submit_all.sh

.PARAMETER Username
    default: jongminm

.PARAMETER Hostname
    default: discovery.usc.edu

.PARAMETER RemoteBase
    default: ~/sparse_kmeans_project
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$ErrorActionPreference = "Stop"

# Define Local Paths
$SimDir = "d:\GitHub\sparse_kmeans\simulations\sim02_unknown_cov_greedy"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Create Remote Directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/sim02_unknown_cov_greedy && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer Simulation Files
Write-Host "Transferring simulation files..." -ForegroundColor Cyan
scp -r "${SimDir}\*" "${Username}@${Hostname}:${RemoteBase}/simulations/sim02_unknown_cov_greedy/"

# 3. Transfer Library Files (only source files, not compiled binaries)
Write-Host "Transferring library files..." -ForegroundColor Cyan
scp "${CodeDir}\*.R" "${CodeDir}\*.cpp" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Compile C++ Libraries on HPC
Write-Host "Compiling C++ libraries on HPC..." -ForegroundColor Cyan
# Use export for environment variables
$compileCmd = "module load rstats/4.5.1 && cd ${RemoteBase}/code_r && export PKG_CXXFLAGS=`$(Rscript -e 'Rcpp:::CxxFlags()') && export PKG_LIBS=`$(Rscript -e 'Rcpp:::LdFlags()') && R CMD SHLIB proj_simplex.cpp selection_utils.cpp 2>&1 | tee compile.log"

ssh "${Username}@${Hostname}" $compileCmd

if ($LASTEXITCODE -ne 0) {
    Write-Error "C++ compilation failed on HPC. Check compile.log"
    exit 1
}

# 5. Convert Line Endings & Submit
Write-Host "Submitting jobs..." -ForegroundColor Cyan
# Clean existing binaries on remote to ensure fresh compilation
# Convert endings for R scripts and shell scripts, then run submit_all.sh
$submitCmd = "cd ${RemoteBase}/simulations/sim02_unknown_cov_greedy && rm -rf logs results *.rds *.out *.err && dos2unix *.sh *.R && chmod +x *.sh && ./submit_all.sh"

ssh "${Username}@${Hostname}" $submitCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Job submitted successfully!" -ForegroundColor Green
}
else {
    Write-Error "Job submission failed."
}
