#!/bin/bash
# Deploys Simulation v3 (Stringent P-val) to HPC.

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

# 1. Prepare and Transfer Files in one SCP
echo -e "${CYAN}Transferring all files...${NC}"

# Create a temporary staging folder on the remote
ssh "${USERNAME}@${HOSTNAME}" "mkdir -p ${REMOTE_BASE}/staging"

# Transfer everything (R and CPP) in one go
scp "${SIM_DIR}/run_sim_v3.R" "${SIM_DIR}/submit_v3.slurm" "${CODE_DIR}/"*.R "${CODE_DIR}/"*.cpp "${USERNAME}@${HOSTNAME}:${REMOTE_BASE}/staging/"

# 2. Organize and Run on Remote in one SSH session
echo -e "${CYAN}Organizing files and submitting job...${NC}"

SUBMIT_CMD="
mkdir -p ${REMOTE_BASE}/simulations/sim_v3_thompson_stringent
mkdir -p ${REMOTE_BASE}/code_r
mv ${REMOTE_BASE}/staging/run_sim_v3.R ${REMOTE_BASE}/simulations/sim_v3_thompson_stringent/
mv ${REMOTE_BASE}/staging/submit_v3.slurm ${REMOTE_BASE}/simulations/sim_v3_thompson_stringent/
mv ${REMOTE_BASE}/staging/*.R ${REMOTE_BASE}/code_r/
mv ${REMOTE_BASE}/staging/*.cpp ${REMOTE_BASE}/code_r/
rmdir ${REMOTE_BASE}/staging

cd ${REMOTE_BASE}/simulations/sim_v3_thompson_stringent
rm -rf logs results *.out *.err
mkdir -p logs results
dos2unix *.slurm *.R
chmod +x *.slurm

cd ${REMOTE_BASE}/code_r
rm -f selection_utils.o selection_utils.so selection_utils.dll Makevars
module purge
module load rstats/4.5.1
# Generate Makevars
cat << 'EOF' > Makevars
PKG_CXXFLAGS += -fopenmp $(shell Rscript -e 'Rcpp:::CxxFlags()')
PKG_LIBS += -fopenmp
EOF

R CMD SHLIB selection_utils.cpp
if [ ! -f selection_utils.so ]; then echo "Error: selection_utils.so not generated"; exit 1; fi

cd ${REMOTE_BASE}/simulations/sim_v3_thompson_stringent
sbatch submit_v3.slurm
"

ssh "${USERNAME}@${HOSTNAME}" "$SUBMIT_CMD"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Job submitted successfully!${NC}"
else
    echo -e "${RED}Job submission failed.${NC}" >&2
    exit 1
fi
