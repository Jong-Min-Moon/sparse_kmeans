#!/bin/bash
#SBATCH --job-name=sim01_greedy
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=1-100%33

# Load R module (C++ already compiled by deploy script on login node)
module purge
module load rstats/4.5.1

# Ensure output directories exist
mkdir -p logs
mkdir -p results_raw/p${P}

# Run replicate — all parameters injected via --export in sbatch call
cd $SLURM_SUBMIT_DIR
Rscript driver.R \
    --job_id $SLURM_ARRAY_TASK_ID \
    --sep    $SEP \
    --p      $P \
    --noise  $NOISE
