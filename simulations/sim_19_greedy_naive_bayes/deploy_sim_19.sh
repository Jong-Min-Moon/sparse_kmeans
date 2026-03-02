#!/bin/bash
set -euo pipefail

# Deploys Simulation 19 (Greedy Naive Bayes) to HPC via macOS Bash.

USERNAME=${1:-"jongminm"}
HOSTNAME=${2:-"discovery.usc.edu"}
REMOTE_BASE=${3:-"~/sparse_kmeans_project"}
SEPARATIONS=(4 5 6)

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
ssh "${USERNAME}@${HOSTNAME}" "mkdir -p ${REMOTE_BASE}/simulations/sim_19_greedy_naive_bayes && mkdir -p ${REMOTE_BASE}/code_r"

# 2. Transfer Simulation Files (Driver, Submit Script)
echo -e "${CYAN}Transferring simulation files...${NC}"
scp "${SIM_DIR}/simulate_sim_19.R" "${SIM_DIR}/submit.sh" "${USERNAME}@${HOSTNAME}:${REMOTE_BASE}/simulations/sim_19_greedy_naive_bayes/"

# 3. Transfer Library Files
echo -e "${CYAN}Transferring library files...${NC}"
scp "${CODE_DIR}/"*.R "${USERNAME}@${HOSTNAME}:${REMOTE_BASE}/code_r/"

# 4. Clean up, convert line endings, and Submit
echo -e "${CYAN}Preparing HPC environment...${NC}"
PREP_CMD="cd ${REMOTE_BASE}/simulations/sim_19_greedy_naive_bayes && \
rm -rf logs results_raw results_aggregated slurm*.out slurm*.err && \
mkdir -p logs/sim_19_greedy_naive_bayes results_raw results_aggregated && \
dos2unix *.sh *.R 2>/dev/null && \
chmod +x *.sh"
ssh "${USERNAME}@${HOSTNAME}" "$PREP_CMD"

echo -e "${CYAN}Submitting SLURM jobs...${NC}"
declare -a SUBMITTED_JOBS=()

for sep in "${SEPARATIONS[@]}"; do
    echo -e "Submitting simulation for sep: ${sep} | Target FDR: 0.4"
    
    SBATCH_CMD="cd ${REMOTE_BASE}/simulations/sim_19_greedy_naive_bayes && \
sbatch --output=logs/sim_19_greedy_naive_bayes/sim_id%a_sep${sep}.out \
--error=logs/sim_19_greedy_naive_bayes/sim_id%a_sep${sep}.err \
--export=ALL,SEP=${sep} submit.sh"
    
    # Run SSH securely evaluating outputs via stdout+stderr
    OUTPUT=$(ssh "${USERNAME}@${HOSTNAME}" "$SBATCH_CMD" 2>&1) || true
    
    # Parse regex tracking Job ID
    if [[ "$OUTPUT" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        JOB_ID="${BASH_REMATCH[1]}"
        echo -e " -> ${GREEN}Successfully submitted Job ID: ${JOB_ID}${NC}"
        SUBMITTED_JOBS+=("Sep: ${sep} | JobID: ${JOB_ID} | Status: Success")
    else
        echo -e " -> ${RED}Failed to submit job for sep ${sep}. Output: ${OUTPUT}${NC}" >&2
        SUBMITTED_JOBS+=("Sep: ${sep} | JobID: N/A | Status: Failed")
    fi
done

echo ""
echo -e "${CYAN}================ Deployment Summary ================${NC}"
for job in "${SUBMITTED_JOBS[@]}"; do
    echo "$job"
done
echo -e "${CYAN}====================================================${NC}"
