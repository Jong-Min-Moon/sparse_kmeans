#!/bin/bash

# run_sim.sh
# Launch knowncov_conda simulations using the correct conda R environment.
#
# Usage:
#   ./run_sim.sh laplace
#   ./run_sim.sh gaussian
#   ./run_sim.sh both

# Colors for output
CYAN='\033[0;36m'
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

SIM=${1:-"both"}
CONDA_ENV="r_legacy_sim"
# Get the absolute path of the directory where the script is located
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

invoke_sim() {
    local script="$1"
    local label="$2"
    
    echo -e "${CYAN}\n==> Running $label simulation...${NC}"
    
    # --no-capture-output streams R's cat() output live to this console
    conda run --no-capture-output -n "$CONDA_ENV" Rscript "$script"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: $label simulation failed (exit code $?).${NC}"
    else
        echo -e "${GREEN}==> $label simulation completed.${NC}"
    fi
}

# Convert SIM to lowercase
SIM_LOWER=$(echo "$SIM" | tr '[:upper:]' '[:lower:]')

case "$SIM_LOWER" in
    "laplace")
        invoke_sim "$ROOT/sim_laplace_knowncov.R" "Laplace"
        ;;
    "gaussian")
        invoke_sim "$ROOT/sim_gaussian_knowncov.R" "Gaussian"
        ;;
    "both"|"")
        invoke_sim "$ROOT/sim_laplace_knowncov.R" "Laplace"
        invoke_sim "$ROOT/sim_gaussian_knowncov.R" "Gaussian"
        ;;
    *)
        echo -e "${RED}Unknown simulation type: $SIM${NC}"
        exit 1
        ;;
esac
