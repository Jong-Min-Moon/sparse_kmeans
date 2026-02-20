# Test block_coordinate_optim_greedy_unknowncov.R
library(MASS)
library(Matrix)
library(glmnet)
library(foreach)
library(doParallel)
library(mclust)

# Source dependencies
source("code_r/sdp_kmeans.R")
source("code_r/ISEE_bicluster.R")
source("code_r/get_intercept_residual_lasso.R")
source("code_r/selection_block_greedy_screening.R")
source("code_r/clustering_block_knowncov.R")
source("code_r/block_coordinate_optim_greedy_unknowncov.R")

if (getDoParWorkers() == 1) registerDoParallel(cores = 4)

cat(paste0(strrep("=", 70), "\n"))
cat("TESTING: block_coordinate_optim_greedy_unknowncov\n")
cat(paste0(strrep("=", 70), "\n\n"))

# =============================================================================
# Generate Test Data
# =============================================================================
set.seed(123)
n <- 100
p <- 50
K <- 2
rho <- 0.45

cat("Generating test data (n=100, p=50, K=2)...\n")

# Precision Matrix (Tridiagonal)
Omega_true <- matrix(0, p, p)
for (i in 1:p) {
    Omega_true[i, i] <- 1
    if (i > 1) Omega_true[i, i - 1] <- rho
    if (i < p) Omega_true[i, i + 1] <- rho
}
Sigma_true <- solve(Omega_true)

# Cluster-specific means (sparse signal)
mu1 <- rep(0, p)
mu1[1:10] <- rnorm(10, mean = 2, sd = 0.5)

mu2 <- rep(0, p)
mu2[1:10] <- rnorm(10, mean = -2, sd = 0.5)

# Generate data
cluster_true <- c(rep(1, n/2), rep(2, n/2))
X <- matrix(0, p, n)
X[, 1:(n/2)] <- t(mvrnorm(n/2, mu1, Sigma_true))
X[, (n/2+1):n] <- t(mvrnorm(n/2, mu2, Sigma_true))

cat("Data dimensions: ", dim(X), "\n")
cat("True cluster sizes: ", table(cluster_true), "\n\n")

# =============================================================================
# Run Algorithm
# =============================================================================
cat(paste0(strrep("=", 70), "\n"))
cat("RUNNING ALGORITHM\n")
cat(paste0(strrep("=", 70), "\n\n"))

tryCatch({
    start_time <- Sys.time()
    
    res <- block_coordinate_optim_greedy_unknowncov(
        X = X,
        K = K,
        n_iter = 5,       # Reduced for quick test
        stable_iter = 2,  # Reduced for quick test
        fdr_level = 0.4
    )
    
    end_time <- Sys.time()
    
    # =============================================================================
    # Report Results
    # =============================================================================
    cat(paste0("\n", strrep("=", 70), "\n"))
    cat("RESULTS\n")
    cat(paste0(strrep("=", 70), "\n\n"))
    
    cat(sprintf("Total iterations: %d\n", res$iter))
    cat(sprintf("Total time: %.2f seconds\n", as.numeric(difftime(end_time, start_time, units = "secs"))))
    cat(sprintf("Selected features: %d\n", sum(res$selected_features)))
    cat(sprintf("Feature indices: %s\n", paste(which(res$selected_features), collapse = ", ")))
    
    # Calculate accuracy
    ari <- mclust::adjustedRandIndex(res$cluster, cluster_true)
    cat(sprintf("\nClustering Accuracy:\n"))
    cat(sprintf("  Adjusted Rand Index: %.4f\n", ari))
    
    cat(sprintf("\nEstimated cluster sizes: %s\n", paste(table(res$cluster), collapse = ", ")))
    cat(sprintf("True cluster sizes: %s\n", paste(table(cluster_true), collapse = ", ")))
    
    cat(paste0("\n", strrep("=", 70), "\n"))
    cat("TEST PASSED: Algorithm ran without errors!\n")
    cat(paste0(strrep("=", 70), "\n"))
    
}, error = function(e) {
    cat(paste0("\n", strrep("=", 70), "\n"))
    cat("TEST FAILED: Error encountered\n")
    cat(paste0(strrep("=", 70), "\n"))
    cat(sprintf("Error message: %s\n", e$message))
    cat(sprintf("Error trace:\n"))
    print(e)
})
