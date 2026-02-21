#!/bin/bash
# Deploys Simulation v3 Oracle ISEE to HPC.

USERNAME=${1:-"jongminm"}
HOSTNAME=${2:-"discovery.usc.edu"}
REMOTE_BASE=${3:-"~/sparse_kmeans_project"}

# Define Local Paths
SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$(dirname "$SIM_DIR")")/code_r"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# 1. Create Remote Directories
echo -e "${CYAN}Creating remote directories...${NC}"
ssh "${USERNAME}@${HOSTNAME}" "mkdir -p ${REMOTE_BASE}/simulations/sim_v4_thompson_fdr && mkdir -p ${REMOTE_BASE}/code_r"

# 2. Transfer Simulation Files
echo -e "${CYAN}Transferring simulation files...${NC}"
scp "${SIM_DIR}/run_sim_v4.R" "${SIM_DIR}/submit_v4.slurm" "${USERNAME}@${HOSTNAME}:${REMOTE_BASE}/simulations/sim_v4_thompson_fdr/"

# 3. Transfer Library Files
echo -e "${CYAN}Transferring library files...${NC}"
scp "${CODE_DIR}/"*.R "${CODE_DIR}/"*.cpp "${USERNAME}@${HOSTNAME}:${REMOTE_BASE}/code_r/"

# 4. Convert Line Endings, Compile C++, & Submit
echo -e "${CYAN}Compiling C++ backend and submitting job...${NC}"
SUBMIT_CMD="cd ${REMOTE_BASE}/simulations/sim_v4_thompson_fdr && \
rm -rf logs results *.out *.err && \
mkdir -p logs results && \
dos2unix *.slurm *.R && \
chmod +x *.slurm && \
cd ${REMOTE_BASE}/code_r && \
rm -f selection_utils.o selection_utils.so selection_utils.dll && \
module purge && module load rstats/4.5.1 && \
export PKG_CXXFLAGS=\"\$(Rscript -e 'cat(Rcpp:::CxxFlags())') -fopenmp\" && \
export PKG_LIBS=\"-fopenmp\" && \
R CMD SHLIB selection_utils.cpp && \
cd ${REMOTE_BASE}/simulations/sim_v4_thompson_fdr && \
sbatch submit_v4.slurm"

ssh "${USERNAME}@${HOSTNAME}" "$SUBMIT_CMD"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Job submitted successfully!${NC}"
else
    echo -e "${RED}Job submission failed.${NC}" >&2
    exit 1
fi
