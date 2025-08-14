#!/bin/bash
#SBATCH --output="/home1/jongminm/sparse_kmeans/experiment/28_07_30_2025/real/real_cheap_rep_29_p.out"
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
 
# Echo job start time and host
echo "Starting job for rep=29 on $(hostname) at $(date)"

# Load necessary modules
module purge
module load legacy/CentOS7
module load matlab/2022a

# Change to base directory
cd "/home1/jongminm/sparse_kmeans/experiment/28_07_30_2025/real"

# Run MATLAB script in batch mode
matlab -batch real_cheap_rep_29_p

# Echo job finish time
echo "Finished job for rep=29 at $(date)"
