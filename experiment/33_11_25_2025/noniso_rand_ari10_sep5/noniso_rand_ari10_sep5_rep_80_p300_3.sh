#!/bin/bash
#SBATCH --output="/home1/jongminm/sparse_kmeans/experiment/33_11_25_2025/noniso_rand_ari10_sep5/noniso_rand_ari10_sep5_rep_80_p300_3.out"
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3:59:59

# Echo job start time and host
echo "Starting job for rep=80 on $(hostname) at $(date)"

# Load necessary modules
module purge
module load legacy/CentOS7
module load matlab/2022a

# Change to base directory
cd "/home1/jongminm/sparse_kmeans/experiment/33_11_25_2025/noniso_rand_ari10_sep5"

# Run MATLAB script in batch mode
matlab -batch noniso_rand_ari10_sep5_rep_80_p300_3

# Echo job finish time
echo "Finished job for rep=80 at $(date)"
