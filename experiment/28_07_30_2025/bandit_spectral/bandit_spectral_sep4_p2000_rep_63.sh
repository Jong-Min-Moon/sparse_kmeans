#!/bin/bash
#SBATCH --job-name=bandit_p2000_rep63 # Job name for easier identification in queue
#SBATCH --output="/home1/jongminm/sparse_kmeans/experiment/28_07_30_2025/bandit_spectral/bandit_spectral_sep4_p2000_rep_63.out"              # Standard output and error log file
#SBATCH --partition=main                   # Specify the partition to use
#SBATCH --nodes=1                          # Request 1 node
#SBATCH --ntasks=1                         # Request 1 task (process)
#SBATCH --cpus-per-task=4                  # Request 8 CPUs per task (for MATLAB's multi-threading)
#SBATCH --mem=8G                           # Request 6 GB of memory
#SBATCH --time=23:59:59                    # Set maximum job run time (HH:MM:SS)

# Echo start time and hostname for logging
echo "Starting job for p=2000, rep=63 on $(hostname) at $(date)"

# Load necessary modules
# Ensure these module commands are correct for your cluster environment
module purge                # Unload all currently loaded modules
module load legacy/CentOS7  # Load specific OS environment if required
module load matlab/2022a    # Load the MATLAB module

# Change to the base directory where the MATLAB script is located
# Use '|| { ... }' for robust error handling if directory change fails
cd "/home1/jongminm/sparse_kmeans/experiment/28_07_30_2025/bandit_spectral" || { echo "Error: Could not change to directory /home1/jongminm/sparse_kmeans/experiment/28_07_30_2025/bandit_spectral. Exiting."; exit 1; }

# Run the MATLAB script in batch mode
# The -batch option runs the specified script and then exits MATLAB
matlab -batch "bandit_spectral_sep4_p2000_rep_63"

# Echo end time for logging
echo "Finished job for p=2000, rep=63 at $(date)"
