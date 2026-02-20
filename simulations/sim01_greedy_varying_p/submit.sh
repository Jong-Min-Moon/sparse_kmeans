#!/bin/bash
#SBATCH --job-name=sim01_greedy_p
#SBATCH --output=sim01_output_%j.txt
#SBATCH --error=sim01_error_%j.txt
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

module purge
# Try loading the module from the template
module load rstats/4.5.1 2>/dev/null || module load R 2>/dev/null || echo "Module load failed"

# Debugging information
echo "Loaded Modules:"
module list
echo "Rscript path: $(which Rscript)"

echo "Starting Simulation Job..."
echo "Current Directory: $(pwd)"

# 1. Compile the C++ Solver
echo "Step 1: Compiling Solver..."
Rscript compile_solver.R

if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi

# 2. Run the Simulation
echo "Step 2: Running Simulation..."
Rscript run_sim01.R

echo "Job Complete."
