#!/bin/bash
#SBATCH --job-name=sim07_perm
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=23:00:00
#SBATCH --output=logs/sim_id%a.out
#SBATCH --error=logs/sim_id%a.err
#SBATCH --array=1-100

module purge
module load rstats/4.5.1

mkdir -p logs
mkdir -p results

# Ensure libraries are available if needed (usually pre-installed on HPC)
# Rscript -e "if (!require(matrixStats)) install.packages('matrixStats')"

# Run Driver
Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --fdr 0.4 --perms 20
