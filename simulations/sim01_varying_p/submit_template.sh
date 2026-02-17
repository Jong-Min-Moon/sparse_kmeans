#!/bin/bash
#SBATCH --job-name=sim01_knowncov
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=23:59:59
#SBATCH --output=logs/sim_p%P%_%a.out
#SBATCH --error=logs/sim_p%P%_%a.err
#SBATCH --array=1-50

module purge
module load rstats/4.5.1

mkdir -p logs
mkdir -p results_p%P%

# %P% and %NITER% will be replaced by the submit_all.sh script
Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --n_iter %NITER% --p %P% --out_dir results_p%P%
