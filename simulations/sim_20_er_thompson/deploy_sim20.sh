#!/bin/bash
set -euo pipefail

# Deploys Simulation 20 (Thompson Sampling on ER data) to HPC.

USERNAME=${1:-"jongminm"}
HOSTNAME=${2:-"discovery.usc.edu"}
REMOTE_BASE=${3:-"~/sparse_kmeans_project"}

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Define Local Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$SCRIPT_DIR"
CODE_DIR="$(cd "${SCRIPT_DIR}/../../code_r" && pwd)"

# 1. Create Remote Directories
echo -e "${CYAN}Creating remote directories...${NC}"
ssh "${USERNAME}@${HOSTNAME}" "mkdir -p ${REMOTE_BASE}/simulations/sim_20_er_thompson && mkdir -p ${REMOTE_BASE}/code_r"

# 2. Transfer Simulation Files
echo -e "${CYAN}Transferring simulation files...${NC}"
scp "${SIM_DIR}/simulation.R" "${SIM_DIR}/submit.sh" "${SIM_DIR}/aggregate.R" "${USERNAME}@${HOSTNAME}:${REMOTE_BASE}/simulations/sim_20_er_thompson/"

# 3. Transfer Library Files
echo -e "${CYAN}Transferring library files...${NC}"
scp "${CODE_DIR}/"*.R "${CODE_DIR}/"*.cpp "${USERNAME}@${HOSTNAME}:${REMOTE_BASE}/code_r/" 2>/dev/null || true

# 4. Clean up, convert line endings, compile C++, and Submit
echo -e "${CYAN}Preparing HPC environment...${NC}"
PREP_CMD="cd ${REMOTE_BASE}/simulations/sim_20_er_thompson && \
rm -rf logs output slurm*.out slurm*.err && \
mkdir -p logs output && \
dos2unix *.sh *.R 2>/dev/null && \
chmod +x *.sh && \
cd ${REMOTE_BASE}/code_r && \
rm -f *.so *.dll *.o Makevars && \
module load rstats/4.5.1 && \
Rscript -e 'writeLines(c(paste0(\"PKG_CXXFLAGS = -fopenmp -I\", system.file(\"include\", package=\"Rcpp\")), \"PKG_LIBS = -fopenmp\"), \"Makevars\")' && \
R CMD SHLIB proj_simplex.cpp && \
rm -f Makevars"
ssh "${USERNAME}@${HOSTNAME}" "$PREP_CMD"

echo -e "${CYAN}Submitting SLURM jobs...${NC}"
declare -a SUBMITTED_JOBS=()

echo -e "Submitting simulation for ER Graph data"

SBATCH_CMD="cd ${REMOTE_BASE}/simulations/sim_20_er_thompson && \
sbatch --output=logs/sim_rep_%a.out \
--error=logs/sim_rep_%a.err \
submit.sh"

# Run SSH securely evaluating outputs via stdout+stderr
OUTPUT=$(ssh "${USERNAME}@${HOSTNAME}" "$SBATCH_CMD" 2>&1) || true

# Parse regex tracking Job ID
if [[ "$OUTPUT" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    JOB_ID="${BASH_REMATCH[1]}"
    echo -e " -> ${GREEN}Successfully submitted Job ID: ${JOB_ID}${NC}"
    SUBMITTED_JOBS+=("JobID: ${JOB_ID} | Status: Success")
else
    echo -e " -> ${RED}Failed to submit job. Output: ${OUTPUT}${NC}" >&2
    SUBMITTED_JOBS+=("JobID: N/A | Status: Failed")
fi

echo ""
echo -e "${CYAN}================ Deployment Summary ================${NC}"
for job in "${SUBMITTED_JOBS[@]}"; do
    echo "$job"
done
echo -e "${CYAN}====================================================${NC}"
