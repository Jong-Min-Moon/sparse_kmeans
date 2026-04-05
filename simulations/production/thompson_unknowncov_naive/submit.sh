#!/bin/bash
#SBATCH --job-name=thompson_unknowncov_naive
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=23:59:00
#SBATCH --array=1-100%50

# Load modules (standard for this HPC env)
module purge
module load rstats/4.5.1 || module load r/4.5.1 || module load r

# Directories
mkdir -p logs
mkdir -p results_raw/p$P

# Run Driver mapped to exported variables
cd $SLURM_SUBMIT_DIR

# Using default Noise of Gaussian if not explicitly passed
if [ -z "$NOISE" ]; then
    NOISE="Gaussian"
fi

Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --sep $SEP --p $P --pval 0.3 --noise $NOISE
