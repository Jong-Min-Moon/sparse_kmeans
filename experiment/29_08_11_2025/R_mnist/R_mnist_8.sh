#!/bin/bash
#SBATCH --job-name=R_R_mnist  # Job name for easier identification
#SBATCH --output="/home1/jongminm/sparse_kmeans/experiment/29_08_11_2025/R_mnist/R_mnist_8.out"             # Standard output and error log file
#SBATCH --partition=main                  # Specify the partition to use (e.g., 'main', 'debug')
#SBATCH --nodes=1                         # Request 1 node
#SBATCH --ntasks=1                        # Request 1 task (process)
#SBATCH --cpus-per-task=8                 # Request 4 CPUs per task (adjust based on R's needs)
#SBATCH --mem=8G                          # Request 8 GB of memory (adjust based on data size) 
#SBATCH --time=1:59:59                    # Set maximum job run time (HH:MM:SS)

module purge
module load rstats/4.3.3
cd "/home1/jongminm/sparse_kmeans/experiment/29_08_11_2025/R_mnist" || { echo "Error: Could not change to directory /home1/jongminm/sparse_kmeans/experiment/29_08_11_2025/R_mnist. Exiting."; exit 1; }

# Run the R script in batch mode
# Pass DIMENSION and REP_NUM as command-line arguments to the R script
Rscript "R_mnist_8.R"    

# Echo end time for logging
echo "Finished R job at $(date)"
