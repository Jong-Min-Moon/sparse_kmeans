#!/bin/bash
#SBATCH --job-name=bandit_p6000_rep43 # Job name for easier identification in queue
#SBATCH --output="/home1/jongminm/sparse_kmeans/experiment/28_08_11_2025/bandit_thinning/bandit_thinning_sep4_p6000_rep_43.out"              # Standard output and error log file
#SBATCH --partition=main                   # Specify the partition to use
#SBATCH --nodes=1                          # Request 1 node
#SBATCH --ntasks=1                         # Request 1 task (process)
#SBATCH --cpus-per-task=3                  # Request 8 CPUs per task (for MATLAB's multi-threading)
#SBATCH --mem=4G                           # Request 6 GB of memory
#SBATCH --time=3:59:59                    # Set maximum job run time (HH:MM:SS)

# Echo start time and hostname for logging
echo "Starting job for p=6000, rep=43 on $(hostname) at $(date)"

# Load necessary modules
# Ensure these module commands are correct for your cluster environment
module purge                # Unload all currently loaded modules
module load legacy/CentOS7  # Load specific OS environment if required
module load matlab/2022a    # Load the MATLAB module

# Change to the base directory where the MATLAB script is located
# Use '|| { ... }' for robust error handling if directory change fails
cd "/home1/jongminm/sparse_kmeans/experiment/28_08_11_2025/bandit_thinning" || { echo "Error: Could not change to directory /home1/jongminm/sparse_kmeans/experiment/28_08_11_2025/bandit_thinning. Exiting."; exit 1; }

# Run the MATLAB script in batch mode
# The -batch option runs the specified script and then exits MATLAB
matlab -batch "bandit_thinning_sep4_p6000_rep_43"

# Echo end time for logging
echo "Finished job for p=6000, rep=43 at $(date)"
