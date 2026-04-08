#!/bin/bash

# Exit on error
set -e

# Setup default parameters
USERNAME="jongminm"
HOSTNAME="discovery.usc.edu"
REMOTE_BASE="~/sparse_kmeans_project"

# Override with command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --username) USERNAME="$2"; shift 2 ;;
    --hostname) HOSTNAME="$2"; shift 2 ;;
    --remote-base) REMOTE_BASE="$2"; shift 2 ;;
    *) echo "Unknown option $1"; exit 1 ;;
  esac
done

SIM_NAME="thompson_unknowncov_naive"
# Use relative directory reference based on the script's location
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_DIR="${REMOTE_BASE}/simulations/production/${SIM_NAME}"

# Define Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${CYAN}Retrieving simulation results and logs from HPC...${NC}"

# Prepare local repository struct internally to absorb outputs natively
# This keeps the results_raw hierarchy.
mkdir -p "${LOCAL_DIR}/results_raw"
mkdir -p "${LOCAL_DIR}/logs"

echo "Syncing results_raw..."
# Using scp -r to pull the entire results_raw directory
scp -r "${USERNAME}@${HOSTNAME}:${REMOTE_DIR}/results_raw/"* "${LOCAL_DIR}/results_raw/"

echo "Syncing logs..."
scp -r "${USERNAME}@${HOSTNAME}:${REMOTE_DIR}/logs/"* "${LOCAL_DIR}/logs/"

echo -e "${GREEN}Sync process complete. You can now execute aggregate_results.R locally.${NC}"
