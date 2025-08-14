#!/bin/bash
#SBATCH --job-name=nmfsl_p4000_rep16 # Job name for easier identification in queue
#SBATCH --output="/home1/jongminm/sparse_kmeans/experiment/30_08_15_2025/nmfsl/nmfsl_4_p4000_rep_16.out"              # Standard output and error log file
#SBATCH --partition=main                   # Specify the partition to use
#SBATCH --nodes=1                          # Request 1 node
#SBATCH --ntasks=1                         # Request 1 task (process)
#SBATCH --cpus-per-task=2                 # Request 8 CPUs per task (for MATLAB's multi-threading)
#SBATCH --mem=2G                           # Request 6 GB of memory
#SBATCH --time=0:59:59                    # Set maximum job run time (HH:MM:SS)

# Echo start time and hostname for logging
echo "Starting job for p=4000, rep=16 on $(hostname) at $(date)"

# Load necessary modules
# Ensure these module commands are correct for your cluster environment
module purge                # Unload all currently loaded modules
module load legacy/CentOS7  # Load specific OS environment if required
module load matlab/2022a    # Load the MATLAB module

# Change to the base directory where the MATLAB script is located
# Use '|| { ... }' for robust error handling if directory change fails
cd "/home1/jongminm/sparse_kmeans/experiment/30_08_15_2025/nmfsl" || { echo "Error: Could not change to directory /home1/jongminm/sparse_kmeans/experiment/30_08_15_2025/nmfsl. Exiting."; exit 1; }

# Run the MATLAB script in batch mode
# The -batch option runs the specified script and then exits MATLAB
matlab -batch "nmfsl_4_p4000_rep_16"

# Echo end time for logging
echo "Finished job for p=4000, rep=16 at $(date)"
