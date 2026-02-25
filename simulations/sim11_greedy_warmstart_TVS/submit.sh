#!/bin/bash
#SBATCH --job-name=sim11_warmstart
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=3G
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
Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --perms 5000 --p_val 0.001 --fdr 0.4 --n_iter_tvs 1000 --n_iter_greedy 200

