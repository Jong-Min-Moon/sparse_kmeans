#!/bin/bash
#SBATCH --job-name=unknowncov_n200_ar1_thompson
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --array=1-100%15

# Load modules (standard for this HPC env)
module purge
module load rstats/4.5.1

# Directories
mkdir -p logs
mkdir -p results_raw/p$P

# Run Driver mapped to exported variables
# Pass the SEPARATION explicitly if provided, otherwise default args inside
Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --sep $SEP --p $P --pval 0.3
