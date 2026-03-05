#!/bin/bash
set -euo pipefail

# Deploys Simulation 21 (Other Methods on ER data) to HPC.

USERNAME=${1:-"jongminm"}
HOSTNAME=${2:-"discovery.usc.edu"}
REMOTE_BASE=${3:-"~/sparse_kmeans_project"}

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Define Local Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$SCRIPT_DIR"
CODE_DIR="$(cd "${SCRIPT_DIR}/../../code_r" && pwd)"

echo -e "${CYAN}Creating remote directories...${NC}"
ssh "${USERNAME}@${HOSTNAME}" "mkdir -p ${REMOTE_BASE}/simulations/sim_21_er_methods && mkdir -p ${REMOTE_BASE}/code_r"

echo -e "${CYAN}Transferring simulation files...${NC}"
scp "${SIM_DIR}/run_simulation.R" "${SIM_DIR}/submit.sh" "${SIM_DIR}/methods_wrapper.R" "${SIM_DIR}/accuracy_utils.R" "${USERNAME}@${HOSTNAME}:${REMOTE_BASE}/simulations/sim_21_er_methods/"

echo -e "${CYAN}Transferring library files...${NC}"
scp "${CODE_DIR}/"*.R "${CODE_DIR}/"*.cpp "${USERNAME}@${HOSTNAME}:${REMOTE_BASE}/code_r/" 2>/dev/null || true

echo -e "${CYAN}Preparing HPC environment...${NC}"
PREP_CMD="cd ${REMOTE_BASE}/simulations/sim_21_er_methods && \
rm -rf logs results results_aggregated*.rds slurm*.out slurm*.err && \
mkdir -p logs results && \
dos2unix *.sh *.R 2>/dev/null && \
chmod +x *.sh && \
cd ${REMOTE_BASE}/code_r && \
rm -f *.so *.dll *.o Makevars && \
module load rstats/4.5.1 && \
Rscript -e 'writeLines(c(paste0(\"PKG_CXXFLAGS = -fopenmp -I\", system.file(\"include\", package=\"Rcpp\")), \"PKG_LIBS = -fopenmp\"), \"Makevars\")' && \
R CMD SHLIB proj_simplex.cpp && \
R CMD SHLIB selection_utils.cpp && \
rm -f Makevars"
ssh "${USERNAME}@${HOSTNAME}" "$PREP_CMD"

echo -e "${CYAN}Submitting SLURM jobs...${NC}"
SBATCH_CMD="cd ${REMOTE_BASE}/simulations/sim_21_er_methods && sbatch submit.sh"

OUTPUT=$(ssh "${USERNAME}@${HOSTNAME}" "$SBATCH_CMD" 2>&1) || true

if [[ "$OUTPUT" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    JOB_ID="${BASH_REMATCH[1]}"
    echo -e " -> ${GREEN}Successfully submitted Job ID: ${JOB_ID}${NC}"
else
    echo -e " -> ${RED}Failed to submit job. Output: ${OUTPUT}${NC}" >&2
fi
