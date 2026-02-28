#!/bin/bash
#SBATCH --job-name=sim14_thomp
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=3G
#SBATCH --time=23:00:00
#SBATCH --array=1-10

module purge
module load rstats/4.5.1

mkdir -p logs
mkdir -p results

# Set C target from SLURM environment or default to 0.5
if [ -z "$C_TARGET" ]; then
    C_TARGET=0.5
fi

# Run Driver
Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --perms 300 --C_val $C_TARGET --n_iter 1000
