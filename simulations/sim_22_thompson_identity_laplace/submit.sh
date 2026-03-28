#!/bin/bash
#SBATCH --job-name=sim22_thompson_identity_laplace
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=23:00:00
#SBATCH --array=1-100%33

# Load modules (standard for this HPC env)
module purge
module load rstats/4.5.1

# Directories
mkdir -p logs
mkdir -p results_raw/p$P

# Run Driver mapped to exported variables
# Pass the SEPARATION explicitly if provided, otherwise default args inside
cd $SLURM_SUBMIT_DIR
Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --sep $SEP --p $P --pval 0.3
