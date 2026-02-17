#!/bin/bash

# Configuration
P_VALUES=(1000 2000 3000 4000 5000)

# Load R module for the login node
module purge
module load rstats/4.5.1 2>/dev/null || module load R 2>/dev/null || echo "Module load failed on login node"

echo "Step 0: Initial Compilation of the C++ Solver..."
# Run compilation once to avoid race conditions in the array jobs
Rscript compile_solver.R

if [ $? -ne 0 ]; then
    echo "Initial compilation failed! Not submitting jobs."
    exit 1
fi

for p in "${P_VALUES[@]}"
do
    echo "Preparing for p=$p..."
    mkdir -p output_p${p}
    mkdir -p logs
    
    echo "Submitting job array for p=$p..."
    
    # Create a temporary job script for this P
    JOB_SCRIPT="submit_p${p}.sh"
    
    # Replace placeholders using sed
    sed "s/%P%/$p/g" submit_template.sh > "$JOB_SCRIPT"
    
    # Submit
    sbatch "$JOB_SCRIPT"
    
    # Clean up (optional)
    # rm "$JOB_SCRIPT"
done

echo "All jobs submitted."
