#!/bin/bash
#SBATCH --output="/home1/jongminm/sparse_kmeans/experiment/31_09_22_2025/mnist_ourmethod/mnist_ourmethod_rep_43.out"
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G

%#SBATCH --time=4:59:59

# Echo job start time and host
echo "Starting job for rep=43 on $(hostname) at $(date)"

# Load necessary modules
module purge
module load legacy/CentOS7
module load matlab/2022a

# Change to base directory
cd "/home1/jongminm/sparse_kmeans/experiment/31_09_22_2025/mnist_ourmethod"

# Run MATLAB script in batch mode
matlab -batch mnist_ourmethod_rep_43

# Echo job finish time
echo "Finished job for rep=43 at $(date)"
