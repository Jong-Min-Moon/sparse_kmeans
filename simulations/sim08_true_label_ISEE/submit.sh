#!/bin/bash
#SBATCH --job-name=sim08_oracle_isee
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=23:00:00
#SBATCH --output=logs/sim_id%a.out
#SBATCH --error=logs/sim_id%a.err
#SBATCH --array=1-100

module load rstats/4.5.1

mkdir -p logs
mkdir -p results

# Run Driver (Oracle ISEE variant)
Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --fdr 0.4 --perms 20
