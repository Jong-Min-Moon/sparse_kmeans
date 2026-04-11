#!/bin/bash
#SBATCH --job-name=unknowncov_n200_chain_greedy_naive
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=1-100%16

# Load R module (C++ already compiled by deploy script on login node)
module purge
module load gcc rstats/4.5.1 || module load r/4.5.1 || module load r

# Ensure output directories exist
mkdir -p logs

# Default noise to Laplace if not set
if [ -z "$NOISE" ]; then
    NOISE="Laplace"
fi

cd $SLURM_SUBMIT_DIR

# Dispatch to the appropriate noise-specific driver
if [ "${NOISE,,}" == "laplace" ]; then
    Rscript driver_laplace.R --job_id $SLURM_ARRAY_TASK_ID --sep $SEP --p $P
else
    Rscript driver_gaussian.R --job_id $SLURM_ARRAY_TASK_ID --sep $SEP --p $P
fi
