#!/bin/bash
#SBATCH --job-name=sim21_er_methods
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=12:00:00

module purge
module load rstats/4.5.1

# We are running this on a single node using parallel R cores
# run_simulation.R natively handles the 1-100 replications locally over doParallel
Rscript run_simulation.R
