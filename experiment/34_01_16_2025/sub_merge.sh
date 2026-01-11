#!/bin/bash
#SBATCH --job-name=merge_bandit
#SBATCH --output=/home1/jongminm/sparse_kmeans/experiment/34_01_16_2025/logs/merge.out
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --mem=4G
module load legacy/CentOS7

module load matlab/2022a
cd "/home1/jongminm/sparse_kmeans/experiment/34_01_16_2025"
matlab -batch "merge_results_to_sqlite('/home1/jongminm/sparse_kmeans/experiment/34_01_16_2025/results_mat', '/home1/jongminm/sparse_kmeans/sparse_kmeans.db', 'bandit_C')"
