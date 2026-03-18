#!/bin/bash
#SBATCH --job-name=sim_cluster_greedy
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=23:00:00
#SBATCH --array=1-100%20

# Load modules (standard for this HPC env)
module purge
module load rstats/4.5.1

# Directories
mkdir -p logs
# Output directory is created by driver_greedy.R as needed, but mkdir -p is safe here too
mkdir -p results_raw/p$P

# Run Driver mapped to exported variables
# P and FDR are expected to be exported by the caller (submit_all.sh)
Rscript driver_greedy.R --job_id $SLURM_ARRAY_TASK_ID --p $P --fdr $FDR
