#!/bin/bash

# Exit on error
set -e

# Setup default parameters
USERNAME="jongminm"
HOSTNAME="discovery.usc.edu"
REMOTE_BASE="~/sparse_kmeans_project"
SEPARATIONS=(6)
DIMENSIONS=(3000 5000)
NOISE="Laplace"

# Process command-line args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --username) USERNAME="$2"; shift 2 ;;
        --hostname) HOSTNAME="$2"; shift 2 ;;
        --remote-base) REMOTE_BASE="$2"; shift 2 ;;
        --separations) 
            # Expecting comma-separated values, e.g., --separations 6,7
            IFS=',' read -r -a SEPARATIONS <<< "$2"
            shift 2 ;;
        --dimensions) 
            # Expecting comma-separated values, e.g., --dimensions 3000,5000
            IFS=',' read -r -a DIMENSIONS <<< "$2"
            shift 2 ;;
        --noise) NOISE="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

SIM_NAME="thompson_unknowncov_naive"
# Use relative directory reference based on the script's location
SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SIM_DIR}/../../../code_r" && pwd)"

# Terminal colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}Creating remote directories...${NC}"
ssh "${USERNAME}@${HOSTNAME}" "mkdir -p ${REMOTE_BASE}/simulations/production/${SIM_NAME} && mkdir -p ${REMOTE_BASE}/code_r"

echo -e "${CYAN}Transferring simulation files...${NC}"
scp "${SIM_DIR}/driver.R" "${SIM_DIR}/submit.sh" "${USERNAME}@${HOSTNAME}:${REMOTE_BASE}/simulations/production/${SIM_NAME}/"

echo -e "${CYAN}Transferring library dependencies natively...${NC}"
scp "${CODE_DIR}/"*.R "${USERNAME}@${HOSTNAME}:${REMOTE_BASE}/code_r/"
scp "${CODE_DIR}/"*.cpp "${USERNAME}@${HOSTNAME}:${REMOTE_BASE}/code_r/"

echo -e "${CYAN}Preparing HPC environment...${NC}"
PREP_CMD="cd ${REMOTE_BASE}/simulations/production/${SIM_NAME} && rm -rf logs results_raw results_aggregated *.out *.err && mkdir -p logs results_raw results_aggregated && dos2unix *.sh *.R 2>/dev/null && chmod +x *.sh"
ssh "${USERNAME}@${HOSTNAME}" "$PREP_CMD"

echo -e "${CYAN}Compiling C++ source files on the HPC...${NC}"
COMPILE_CMD="export LC_ALL=C; module purge; module load gcc rstats/4.5.1 || module load r/4.5.1 || module load r; cd ${REMOTE_BASE}/code_r && R CMD SHLIB selection_utils.cpp -o selection_utils.so && R CMD SHLIB proj_simplex.cpp -o proj_simplex.so"
ssh "${USERNAME}@${HOSTNAME}" "$COMPILE_CMD"
echo -e "${GREEN}Compilation complete.${NC}"

echo -e "${CYAN}Submitting simulation array jobs to the Slurm scheduler...${NC}"
declare -a SUBMITTED_JOBS

for SEP in "${SEPARATIONS[@]}"; do
    for P in "${DIMENSIONS[@]}"; do
        echo "Submitting simulation for sep: ${SEP}, p: ${P}, noise: ${NOISE}..."
        
        LOG_PATTERN="logs/sim_id%a_sep${SEP}_p${P}_${NOISE}"
        SBATCH_CMD="cd ${REMOTE_BASE}/simulations/production/${SIM_NAME} && sbatch --output=${LOG_PATTERN}.out --error=${LOG_PATTERN}.err --export=ALL,SEP=${SEP},P=${P},NOISE=${NOISE} submit.sh"
        
        MAX_RETRIES=5
        RETRY_WAIT=5
        ATTEMPT=0
        SUCCESS=false
        JOB_ID="N/A"
        
        while [[ $ATTEMPT -lt $MAX_RETRIES && "$SUCCESS" == false ]]; do
            # Disable set -e temporarily to allow exit code checks
            set +e
            OUTPUT=$(ssh "${USERNAME}@${HOSTNAME}" "$SBATCH_CMD" 2>&1)
            EXIT_CODE=$?
            set -e
            
            if [[ $EXIT_CODE -eq 0 && "$OUTPUT" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
                SUCCESS=true
                JOB_ID="${BASH_REMATCH[1]}"
            else
                ATTEMPT=$((ATTEMPT+1))
                if [[ $ATTEMPT -lt $MAX_RETRIES ]]; then
                    echo -e "${YELLOW}SSH timeout or cluster rate limit hit. Retrying $ATTEMPT/$MAX_RETRIES in $RETRY_WAIT seconds...${NC}"
                    sleep $RETRY_WAIT
                fi
            fi
        done
        
        if [[ "$SUCCESS" == true ]]; then
            echo -e "${GREEN} -> Successfully submitted Array Job ID: ${JOB_ID}${NC}"
            SUBMITTED_JOBS+=("Sep: ${SEP} | P: ${P} | Noise: ${NOISE} | JobID: ${JOB_ID} | Status: Success")
        else
            echo -e "${RED} -> Failed to submit job for separation condition ${SEP}, dimension ${P} after $MAX_RETRIES attempts. Output: $OUTPUT${NC}"
            SUBMITTED_JOBS+=("Sep: ${SEP} | P: ${P} | Noise: ${NOISE} | JobID: N/A | Status: Failed")
        fi
        
        # Add a brief pause to prevent SSH connection rate limiting/timeouts on the head node
        sleep 15
    done
done

echo -e "\n${CYAN}================ Deployment Summary ================${NC}"
for JOB in "${SUBMITTED_JOBS[@]}"; do
    echo "$JOB"
done
echo "===================================================="
