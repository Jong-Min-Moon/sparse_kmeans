#!/usr/bin/env bash

# Retrieves Simulation 07 results from the HPC cluster.

USERNAME=${1:-"jongminm"}
HOSTNAME=${2:-"discovery.usc.edu"}
REMOTE_BASE=${3:-"~/sparse_kmeans_project"}

# Get the directory where the script is located
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REMOTE_SIM_DIR="${REMOTE_BASE}/simulations/sim12_greedy_warmstart_TVS_v2"
REMOTE_RESULTS_DIR="${REMOTE_SIM_DIR}/results"
LOCAL_OUTPUT_DIR="${LOCAL_DIR}/output"

# Colors
CYAN='\033[0;36m'
GRAY='\033[0;90m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}Retrieving results from ${USERNAME}@${HOSTNAME}:${REMOTE_RESULTS_DIR} ...${NC}"
echo -e "${CYAN}Local destination: ${LOCAL_OUTPUT_DIR}/${NC}"

# Create local output directory
mkdir -p "${LOCAL_OUTPUT_DIR}"

# Copy all results
echo -e "${GRAY}Copying results directory...${NC}"
scp -r "${USERNAME}@${HOSTNAME}:${REMOTE_RESULTS_DIR}" "${LOCAL_DIR}/"
EXIT_CODE=$?

# Rename to output
if [ -d "${LOCAL_DIR}/results" ]; then
    rm -rf "${LOCAL_OUTPUT_DIR}"
    mv "${LOCAL_DIR}/results" "${LOCAL_OUTPUT_DIR}"
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}Results retrieved successfully!${NC}"
    echo -e "${GREEN}Results saved to: ${LOCAL_OUTPUT_DIR}${NC}"
else
    echo -e "${RED}Failed to retrieve results.${NC}" >&2
    exit 1
fi
