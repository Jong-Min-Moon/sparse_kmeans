#!/bin/bash
#SBATCH --job-name=sim14_thomp
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=21-100

module purge
module load rstats/4.5.1

mkdir -p logs
mkdir -p results_raw

# Run Driver mapped to exported variables
Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --sep $SEP --pval $PVAL --n_step_admm 5000
