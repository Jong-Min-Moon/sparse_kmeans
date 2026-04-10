#!/bin/bash
#SBATCH --job-name=sim07_perm
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=3G
#SBATCH --time=23:00:00
#SBATCH --array=1-10

module purge
module load rstats/4.5.1

mkdir -p logs
mkdir -p results

# Ensure libraries are available if needed (usually pre-installed on HPC)
# Rscript -e "if (!require(matrixStats)) install.packages('matrixStats')"

# Set FDR target from SLURM environment or default to 0.4
if [ -z "$FDR_TARGET" ]; then
    FDR_TARGET=0.4
fi

# Format for log output explicitly:
# We map 0.4 -> 0p4
FDR_STR=$(echo $FDR_TARGET | sed 's/\./p/g')

# Run Driver
Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --fdr $FDR_TARGET --perms 5000
