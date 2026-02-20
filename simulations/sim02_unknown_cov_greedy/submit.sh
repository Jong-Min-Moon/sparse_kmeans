#!/bin/bash
#SBATCH --job-name=sim02_unknowncov
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=5G
#SBATCH --time=23:00:00
#SBATCH --output=logs/sim_id%a.out
#SBATCH --error=logs/sim_id%a.err
#SBATCH --array=1-100

module purge
module load rstats/4.5.1

mkdir -p logs
mkdir -p results

Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID
