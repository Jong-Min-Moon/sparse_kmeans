#!/bin/bash
#SBATCH --job-name=sim_v10
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3G
#SBATCH --time=23:00:00
#SBATCH --output=logs/sim_id%a.out
#SBATCH --error=logs/sim_id%a.err
#SBATCH --array=1-10

module purge
module load rstats/4.5.1

mkdir -p logs
mkdir -p results

# Run Driver
Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --p_val 0.001 --perms 10000 --C 0.3
