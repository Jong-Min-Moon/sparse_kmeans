#!/bin/bash
#SBATCH --job-name=unknowncov_n200_ar1_thompson
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3G
#SBATCH --time=23:00:00
#SBATCH --array=1-100%15

# Load modules (standard for this HPC env)
module purge
module load rstats/4.5.1

# Directories
mkdir -p logs
mkdir -p results_raw/p$P

# Run Driver conditionally mapped to exported variables
if [ -z "$NOISE" ]; then
    NOISE="Gaussian"
fi

if [ "${NOISE,,}" == "laplace" ]; then
    Rscript sim_laplace.R --job_id $SLURM_ARRAY_TASK_ID --sep $SEP --p $P --pval 0.3
else
    Rscript sim_gaussian.R --job_id $SLURM_ARRAY_TASK_ID --sep $SEP --p $P --pval 0.3
fi
