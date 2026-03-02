#!/bin/bash
#SBATCH --job-name=sim15_witten_arias
#SBATCH --output=logs/sim15_%A_%a.out
#SBATCH --error=logs/sim15_%A_%a.err
#SBATCH --array=1-100%20
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=debug

# Parse Separation Level (default to 4 if not provided)
SEP=${1:-4}

echo "Starting job array index $SLURM_ARRAY_TASK_ID with Separation Level $SEP"

# Load R module
module load R/4.1.2

# Create directories
mkdir -p logs results

# Run the simulation driver
Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --sep $SEP
