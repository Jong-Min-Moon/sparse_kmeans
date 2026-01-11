#!/bin/bash
#SBATCH --job-name=bandit_sim
#SBATCH --output=/home1/jongminm/sparse_kmeans/experiment/34_01_16_2025/logs/job_%a.out
#SBATCH --error=/home1/jongminm/sparse_kmeans/experiment/34_01_16_2025/logs/job_%a.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=23:59:59
#SBATCH --array=1-125%50

module load legacy/CentOS7
module load matlab/2022a

cd "/home1/jongminm/sparse_kmeans/experiment/34_01_16_2025"
matlab -batch "run_bandit_task(${SLURM_ARRAY_TASK_ID}, '/home1/jongminm/sparse_kmeans/experiment/34_01_16_2025/param_list.txt', '/home1/jongminm/sparse_kmeans/experiment/34_01_16_2025/results_mat')"
