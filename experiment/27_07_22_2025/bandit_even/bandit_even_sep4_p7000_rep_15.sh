#!/bin/bash
#SBATCH --job-name=bandit_p7000_rep15 # Job name for easier identification in queue
#SBATCH --output="/home1/jongminm/sparse_kmeans/experiment/27_07_22_2025/bandit_even/bandit_even_sep4_p7000_rep_15.out"              # Standard output and error log file
#SBATCH --partition=main                   # Specify the partition to use
#SBATCH --nodes=1                          # Request 1 node
#SBATCH --ntasks=1                         # Request 1 task (process)
#SBATCH --cpus-per-task=4                  # Request 8 CPUs per task (for MATLAB's multi-threading)
#SBATCH --mem=6G                           # Request 6 GB of memory
#SBATCH --time=23:59:59                    # Set maximum job run time (HH:MM:SS)

# Echo start time and hostname for logging
echo "Starting job for p=7000, rep=15 on $(hostname) at $(date)"

# Load necessary modules
# Ensure these module commands are correct for your cluster environment
module purge                # Unload all currently loaded modules
module load legacy/CentOS7  # Load specific OS environment if required
module load matlab/2022a    # Load the MATLAB module

# Change to the base directory where the MATLAB script is located
# Use '|| { ... }' for robust error handling if directory change fails
cd "/home1/jongminm/sparse_kmeans/experiment/27_07_22_2025/bandit_even" || { echo "Error: Could not change to directory /home1/jongminm/sparse_kmeans/experiment/27_07_22_2025/bandit_even. Exiting."; exit 1; }

# Run the MATLAB script in batch mode
# The -batch option runs the specified script and then exits MATLAB
matlab -batch "bandit_even_sep4_p7000_rep_15"

# Echo end time for logging
echo "Finished job for p=7000, rep=15 at $(date)"
