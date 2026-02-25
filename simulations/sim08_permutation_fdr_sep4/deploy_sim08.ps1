
<#
.SYNOPSIS
    Deploys Simulation 08 (Permutation FDR 0.4, Sep=4) to HPC.
#>

param(
    [string]$Username = "jongminm",
    [string]$Hostname = "discovery.usc.edu",
    [string]$RemoteBase = "~/sparse_kmeans_project"
)

$ErrorActionPreference = "Stop"

# Define Local Paths
$SimDir = "d:\GitHub\sparse_kmeans\simulations\sim08_permutation_fdr_sep4"
$CodeDir = "d:\GitHub\sparse_kmeans\code_r"

# 1. Create Remote Directories
Write-Host "Creating remote directories..." -ForegroundColor Cyan
ssh "${Username}@${Hostname}" "mkdir -p ${RemoteBase}/simulations/sim08_permutation_fdr_sep4 && mkdir -p ${RemoteBase}/code_r"

# 2. Transfer Simulation Files (Driver, Submit Script)
Write-Host "Transferring simulation files..." -ForegroundColor Cyan
scp "${SimDir}\driver.R" "${SimDir}\submit.sh" "${Username}@${Hostname}:${RemoteBase}/simulations/sim08_permutation_fdr_sep4/"

# 3. Transfer Library Files
Write-Host "Transferring library files..." -ForegroundColor Cyan
# Crucial: Ensure the new block_coordinate_optim_greedy_unknowncov_SAM.R is transferred
scp "${CodeDir}\*.R" "${Username}@${Hostname}:${RemoteBase}/code_r/"
# Also transfer C++ source for Rcpp compilation
scp "${CodeDir}\selection_utils.cpp" "${Username}@${Hostname}:${RemoteBase}/code_r/"

# 4. Compile C++ Backend on Remote
# 4. Compile C++ Backend on Remote
Write-Host "Compiling C++ backend on remote..." -ForegroundColor Cyan
# Fetch Rcpp flags dynamically
$compileCmd = "cd ${RemoteBase}/code_r && export PKG_CXXFLAGS=`$(Rscript -e 'cat(Rcpp:::CxxFlags())') && export PKG_LIBS=`$(Rscript -e 'cat(Rcpp:::LdFlags())') && R CMD SHLIB selection_utils.cpp"
ssh "${Username}@${Hostname}" "module load rstats/4.5.1 && $compileCmd"

# 5. Convert Line Endings & Submit
Write-Host "Submitting job..." -ForegroundColor Cyan
$submitCmd = "cd ${RemoteBase}/simulations/sim08_permutation_fdr_sep4 && rm -rf logs results *.out *.err && dos2unix *.sh *.R && chmod +x *.sh && sbatch submit.sh"
ssh "${Username}@${Hostname}" $submitCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Job submitted successfully!" -ForegroundColor Green
}
else {
    Write-Error "Job submission failed."
}
