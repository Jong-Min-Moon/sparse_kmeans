#!/bin/bash
#SBATCH --output="/home1/jongminm/sparse_kmeans/experiment/28_07_30_2025/noniso_t8/noniso_t8_rep_41_p300.out"
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
 
# Echo job start time and host
echo "Starting job for rep=41 on $(hostname) at $(date)"

# Load necessary modules
module purge
module load legacy/CentOS7
module load matlab/2022a

# Change to base directory
cd "/home1/jongminm/sparse_kmeans/experiment/28_07_30_2025/noniso_t8"

# Run MATLAB script in batch mode
matlab -batch noniso_t8_rep_41_p300

# Echo job finish time
echo "Finished job for rep=41 at $(date)"
