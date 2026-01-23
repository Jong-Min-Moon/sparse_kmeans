#!/bin/bash
#SBATCH --job-name=bandit_adaptiveC
#SBATCH --output=/home1/jongminm/sparse_kmeans/experiment/35_01_23_2026/logs/job_%a.out
#SBATCH --error=/home1/jongminm/sparse_kmeans/experiment/35_01_23_2026/logs/job_%a.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=1-100%20

module load legacy/CentOS7
module load matlab/2022a

cd "/home1/jongminm/sparse_kmeans/experiment/35_01_23_2026"
matlab -batch "run_bandit_adaptiveC_task(${SLURM_ARRAY_TASK_ID}, 0.1, 0.6, 8000, 1000, '/home1/jongminm/sparse_kmeans/experiment/35_01_23_2026/results/adaptiveC')"
