#!/bin/bash
#SBATCH --output="/home1/jongminm/sparse_kmeans/experiment/28_07_30_2025/noniso_t6/noniso_t6_rep_128_p300.out"
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
 
# Echo job start time and host
echo "Starting job for rep=128 on $(hostname) at $(date)"

# Load necessary modules
module purge
module load legacy/CentOS7
module load matlab/2022a

# Change to base directory
cd "/home1/jongminm/sparse_kmeans/experiment/28_07_30_2025/noniso_t6"

# Run MATLAB script in batch mode
matlab -batch noniso_t6_rep_128_p300

# Echo job finish time
echo "Finished job for rep=128 at $(date)"
