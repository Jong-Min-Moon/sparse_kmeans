#!/bin/bash
#SBATCH --job-name=sim22_thompson_identity
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=23:00:00
#SBATCH --array=1-100%10

# Load modules (standard for this HPC env)
module purge
module load rstats/4.5.1

# Directories
mkdir -p logs
mkdir -p results_raw/p$P

# Run Driver mapped to exported variables
# Pass the SEPARATION explicitly if provided, otherwise default args inside
Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --sep $SEP --p $P
