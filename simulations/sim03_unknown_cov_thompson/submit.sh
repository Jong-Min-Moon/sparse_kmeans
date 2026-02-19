#!/bin/bash
#SBATCH --job-name=sim03_thompson
#SBATCH --output=logs/sim_%a.out
#SBATCH --error=logs/sim_%a.err
#SBATCH --array=1-100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=main

module load rstats/4.5.1

# Run the simulation script
Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID
