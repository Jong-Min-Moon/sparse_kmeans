#!/bin/bash
#SBATCH --job-name=unknowncov_n500_chain_competitors_modern
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --array=1-100%16

# Load modules (standard for this HPC env)
module purge
module load gcc rstats/4.5.1 || module load r/4.5.1 || module load r

# Directories
mkdir -p logs

# Run Driver Based on NOISE Export
if [ -z "$NOISE" ]; then
    NOISE="laplace"
fi

cd $SLURM_SUBMIT_DIR

# Using lowercase NOISE to standardise matching, but script expects properly capitalised output via R script if needed
if [ "${NOISE,,}" == "laplace" ]; then
    Rscript sim_laplace.R --job_id $SLURM_ARRAY_TASK_ID --sep $SEP --p $P
else
    Rscript sim_gaussian.R --job_id $SLURM_ARRAY_TASK_ID --sep $SEP --p $P
fi
