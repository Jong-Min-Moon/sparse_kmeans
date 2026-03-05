#!/bin/bash
#SBATCH --job-name=sim20_er
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=1-100

module purge
module load rstats/4.5.1

mkdir -p logs
mkdir -p output

# Run simulation
Rscript simulation.R $SLURM_ARRAY_TASK_ID output
