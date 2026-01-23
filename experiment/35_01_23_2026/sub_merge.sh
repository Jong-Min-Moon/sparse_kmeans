#!/bin/bash
#SBATCH --job-name=merge_bandit_adaptiveC
#SBATCH --output=/home1/jongminm/sparse_kmeans/experiment/35_01_23_2026/logs/merge.out
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --mem=4G

module load legacy/CentOS7
module load matlab/2022a

cd "/home1/jongminm/sparse_kmeans/experiment/35_01_23_2026"
matlab -batch "merge_adaptiveC_results('/home1/jongminm/sparse_kmeans/experiment/35_01_23_2026/results/adaptiveC')"
