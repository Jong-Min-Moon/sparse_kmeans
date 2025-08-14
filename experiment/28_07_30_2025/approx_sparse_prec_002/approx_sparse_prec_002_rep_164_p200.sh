#!/bin/bash
#SBATCH --output="/home1/jongminm/sparse_kmeans/experiment/28_07_30_2025/approx_sparse_prec_002/approx_sparse_prec_002_rep_164_p200.out"
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3:59:59

# Echo job start time and host
echo "Starting job for rep=164 on $(hostname) at $(date)"

# Load necessary modules
module purge
module load legacy/CentOS7
module load matlab/2022a

# Change to base directory
cd "/home1/jongminm/sparse_kmeans/experiment/28_07_30_2025/approx_sparse_prec_002"

# Run MATLAB script in batch mode
matlab -batch approx_sparse_prec_002_rep_164_p200

# Echo job finish time
echo "Finished job for rep=164 at $(date)"
