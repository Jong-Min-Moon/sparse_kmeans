
<#
.SYNOPSIS
    Deploys Simulation 03 (Thompson Sampling, 100 reps) to HPC.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$ErrorActionPreference = "Stop"

# Define Local Paths
$SimDir = "d:\GitHub\sparse_kmeans\simulations\sim03_unknown_cov_thompson"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Create Remote Directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/sim03_unknown_cov_thompson && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer Simulation Files
Write-Host "Transferring simulation files..." -ForegroundColor Cyan
scp -r "${SimDir}\*" "${Username}@${Hostname}:${RemoteBase}/simulations/sim03_unknown_cov_thompson/"

# 3. Transfer Library Files
Write-Host "Transferring library files..." -ForegroundColor Cyan
scp "${CodeDir}\*.R" "${CodeDir}\*.cpp" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Compile C++ Libraries on HPC (if needed, though already done for sim02)
Write-Host "Compiling C++ libraries on HPC..." -ForegroundColor Cyan
$compileCmd = "module load rstats/4.5.1 && cd ${RemoteBase}/code_r && export PKG_CXXFLAGS=`$(Rscript -e 'Rcpp:::CxxFlags()') && export PKG_LIBS=`$(Rscript -e 'Rcpp:::LdFlags()') && R CMD SHLIB proj_simplex.cpp selection_utils.cpp 2>&1 | tee compile.log"
ssh "${Username}@${Hostname}" $compileCmd

if ($LASTEXITCODE -ne 0) {
    Write-Error "C++ compilation failed on HPC."
    exit 1
}

# 5. Convert Line Endings & Submit
Write-Host "Submitting jobs..." -ForegroundColor Cyan
$submitCmd = "cd ${RemoteBase}/simulations/sim03_unknown_cov_thompson && rm -rf logs results *.rds *.out *.err && dos2unix *.sh *.R && chmod +x *.sh && ./submit_all.sh"
ssh "${Username}@${Hostname}" $submitCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Job submitted successfully!" -ForegroundColor Green
}
else {
    Write-Error "Job submission failed."
}
