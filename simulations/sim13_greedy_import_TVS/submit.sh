#!/bin/bash
#SBATCH --job-name=sim13_import
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=3G
#SBATCH --time=23:00:00
#SBATCH --array=1-10

module purge
module load rstats/4.5.1

mkdir -p logs
mkdir -p results

# Set FDR target from SLURM environment or default to 0.4
if [ -z "$FDR_TARGET" ]; then
    FDR_TARGET=0.4
fi

# Run Driver
Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --perms 5000 --p_val 0.001 --fdr $FDR_TARGET --n_iter_tvs 1000 --n_iter_greedy 200

