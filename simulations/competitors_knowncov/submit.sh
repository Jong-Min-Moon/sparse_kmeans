#!/bin/bash
#SBATCH --job-name=competitors_knowncov
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --array=1-2

# Load modules (standard for this HPC env)
module purge
module load rstats/4.5.1 || module load r/4.5.1 || module load r

# Directories
mkdir -p logs
# Output subdirectories are handled by hpc_driver.R

# Run Driver
# Parameters: SEP, P, NOISE, METHODS are exported by sbatch --export
# Default METHODS to "witten,arias,ifpca,cvs" if not provided
if [ -z "$METHODS" ]; then
    METHODS="witten,arias,ifpca,cvs"
fi

cd $SLURM_SUBMIT_DIR
Rscript hpc_driver.R --job_id $SLURM_ARRAY_TASK_ID --sep $SEP --p $P --noise $NOISE --methods $METHODS
