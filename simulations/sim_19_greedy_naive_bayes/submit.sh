#!/bin/bash
#SBATCH --job-name=sim19_grdy
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=1-30

module purge
module load rstats/4.5.1

mkdir -p logs/sim_19_greedy_naive_bayes
mkdir -p results_raw

# Run Driver mapped to exported variables handled by outer .ps1 wrapper
Rscript simulate_sim_19.R --job_id $SLURM_ARRAY_TASK_ID --sep $SEP --fdr 0.4 --n_step_admm 5000
