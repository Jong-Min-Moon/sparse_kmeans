#!/bin/bash
#SBATCH --job-name=sparse_kmeans_sim
#SBATCH --partition=main              # CARC partition 'main'
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1             # Adjust if parallelizing R code
#SBATCH --mem=4G                      # Adjust based on needs
#SBATCH --time=01:00:00               # 1 hour max
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --array=1-100                 # Run 100 jobs

# Load R module (CARC specific)
module purge
module load rstats/4.5.1

# Create logs directory if not exists
mkdir -p logs

# Run Simulation Driver
# $SLURM_ARRAY_TASK_ID is passed as the job ID
Rscript simulation_driver.R --job_id $SLURM_ARRAY_TASK_ID --n_iter 100 --out_dir results
