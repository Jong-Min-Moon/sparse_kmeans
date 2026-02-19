# driver_matlab_parity.R
# Replicate the user's MATLAB script logic in R

library(MASS)
library(mclust)
library(Matrix)
library(foreach)
library(doParallel)
library(glmnet)

# Source all components
source("../../code_r/sparse_symmetric_data_generator.R")
source("../../code_r/block_coordinate_optim_deterministic_unknowncov.R")
source("../../code_r/ESSC.R")
source("../../code_r/ISEE_residual_lasso.R")
source("../../code_r/get_intercept_residual_lasso.R")
source("../../code_r/get_cov_small.R")
source("../../code_r/ISEE_bicluster.R")
source("../../code_r/clustering_block_knowncov.R")
source("../../code_r/sdp_kmeans.R")
source("../../code_r/get_cluster_acc.R")
source("../../code_r/utils.R")

# Parallel setup (matching MATLAB parpool)
ncores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")) - 1
if (ncores < 1) ncores <- parallel::detectCores() - 1
if (ncores < 1) ncores <- 1

cl <- makeCluster(ncores)
registerDoParallel(cl)
cat(sprintf("Using %d cores for parallel ISEE.\n", ncores))

# Experimental Settings
iternum_max <- 100
rho_param <- 20
dimension <- 100
separation <- 3
sample_size <- 500
rho_val <- rho_param / 100
precision_sparsity <- 2
support <- 1:10
flip <- FALSE

cat("--- MATLAB Parity Simulation (p=100, n=500, sep=3, rho=0.2) ---\n")

# 1. Initialize Generator
generator <- sparse_symmetric_data_generator(
    support = support,
    separation = separation,
    dimension = dimension,
    precision_sparsity = precision_sparsity,
    conditional_correlation = rho_val,
    flip = flip
)

# 2. Generate Data (RNG seed logic is handled inside for reproducibility)
data_res <- generate_data_from_generator(generator, sample_size, seed = 42)
X <- data_res$X
cluster_true <- data_res$labels

# 3. Run Optimization (Iterative ISEE-KMeans)
# MATLAB: cluster_estimte = ISEE_kmeans_noisy(x_noisy, 2, 100, true)
# Our R version follows exactly that logic in block_coordinate_optim_deterministic_unknowncov.
res <- block_coordinate_optim_deterministic_unknowncov(
    X = X,
    K = 2,
    n_iter = 50,
    stable_iter = 5,
    true_labels = cluster_true,
    ari_consecutive_stop = 10
)

# 4. Final Accuracy
acc_final <- get_cluster_acc(res$cluster, cluster_true) # Changed function and variable name
cat(sprintf("\nFinal Balanced Accuracy (acc): %.4f\n", acc_final)) # Changed variable name
cat(sprintf("Final ARI: %.4f\n", mclust::adjustedRandIndex(res$cluster, cluster_true)))
cat(sprintf("Features Selected: %d\n", length(res$s_hat)))
cat(sprintf("Signal Captured: %d/10\n", sum(res$s_hat %in% support)))

stopCluster(cl)
