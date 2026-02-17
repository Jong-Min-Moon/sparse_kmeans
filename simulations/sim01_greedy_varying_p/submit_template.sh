#!/bin/bash
#SBATCH --job-name=sim01_p%P%
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sim_p%P%_%a.out
#SBATCH --error=logs/sim_p%P%_%a.err
#SBATCH --array=1-100

module purge
module load rstats/4.5.1 2>/dev/null || module load R 2>/dev/null || echo "Module load failed"

mkdir -p logs
mkdir -p output_p%P%

# %P% will be replaced by submit_all.sh
# %a is the array index
Rscript run_sim01.R --p %P% --rep $SLURM_ARRAY_TASK_ID --out_dir output_p%P%
