# ---------------------------------------------------------
# Single Simulation Run: Unknown Covariance (Greedy)
# ---------------------------------------------------------
library(methods)
library(MASS)
library(mclust)
library(Matrix)
library(foreach)
library(doParallel)
library(glmnet)

# Source Code
source("code_r/block_coordinate_optim_greedy_unknowncov.R")
source("code_r/ISEE_bicluster.R")
source("code_r/get_intercept_residual_lasso.R")
source("code_r/clustering_block_knowncov.R")
source("code_r/selection_block_greedy_screening.R")
source("code_r/sdp_kmeans.R")
source("code_r/utils.R")

set.seed(2025)

# ---------------------------------------------------------
# Data Generation Parameters
# ---------------------------------------------------------
p <- 400
n <- 500
K <- 2
rho <- 0.45

# 1. Precision Matrix Omega (Tridiagonal)
# Omega_ii = 1, Omega_i,i+1 = rho
cat("Generating Precision Matrix Omega...\n")
Omega <- matrix(0, p, p)
for (i in 1:p) {
    Omega[i, i] <- 1
    if (i > 1) Omega[i, i - 1] <- rho
    if (i < p) Omega[i, i + 1] <- rho
}

# Verify positive definiteness
eigen_vals <- eigen(Omega, only.values = TRUE)$values
if (min(eigen_vals) <= 0) {
    stop("Omega is not positive definite!")
}

# Covariance Sigma
Sigma <- solve(Omega)

# 2. Signal Generation
# S_0 = 1:10
S_0 <- 1:10
cat("Generating Signal...\n")

# Target: || Omega * (mu1 - mu2) ||^2 = 9 (3^2)
# Assumption: Delta_mu is constant delta on S_0
# Let Delta_mu = delta * v, where v_j = 1 if j in S0, 0 else.
# Target = delta^2 * || Omega * v ||^2 = 9
# delta = sqrt( 9 / || Omega * v ||^2 )

v <- rep(0, p)
v[S_0] <- 1

Omega_v <- Omega %*% v
norm_sq_Omega_v <- sum(Omega_v^2)
delta <- sqrt(9 / norm_sq_Omega_v)

cat(sprintf("Calculated delta per active feature: %.4f\n", delta))

mu_diff <- v * delta
# Centered around 0
mu1 <- mu_diff / 2
mu2 <- -mu_diff / 2

# Verification
actual_norm_sq <- sum((Omega %*% (mu1 - mu2))^2)
cat(sprintf("Signal Strength: || Omega*DeltaMu ||^2 = %.4f (Target: 9.0)\n", actual_norm_sq))

# 3. Generate Data
cat("Generating Data X...\n")
n_c <- n / 2
X1 <- mvrnorm(n_c, mu1, Sigma)
X2 <- mvrnorm(n_c, mu2, Sigma)
X <- t(rbind(X1, X2)) # p x n
true_labels <- c(rep(1, n_c), rep(2, n_c))

# ---------------------------------------------------------
# Run Algorithm
# ---------------------------------------------------------
cat("Running block_coordinate_optim_greedy_unknowncov...\n")

# Register Parallel (if not already)
if (getDoParWorkers() == 1) {
    if (requireNamespace("doParallel", quietly = TRUE)) {
        doParallel::registerDoParallel(cores = 2)
    } else {
        registerDoSEQ()
    }
}

start_time <- Sys.time()
res <- block_coordinate_optim_greedy_unknowncov(X, K = 2, n_iter = 10, stable_iter = 3, fdr_level = 0.1)
end_time <- Sys.time()

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
cat("\n--- Results ---\n")
runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat(sprintf("Runtime: %.2f seconds\n", runtime))

# Clustering Accuracy (ARI)
ari <- adjustedRandIndex(res$cluster, true_labels)
cat(sprintf("Adjusted Rand Index (ARI): %.4f\n", ari))

# Variable Selection
selected_indices <- which(res$selected_features) # Assuming output has this or we need to extract from last iteration
# The function returns "cluster", "iter", "rand_vec".
# Wait, looking at the code, it returns: list(cluster = cluster_est_now, iter = iternum, rand_vec = rand_vec, time = total_time)
# It DOES NOT Return the selected features?
# I need to check the code again. It seems it might be missing that return.
# WARNING: block_coordinate_optim_greedy_unknowncov might vary from Thompson which returns more.
# Let's inspect variable selection by looking at the loop in the function.
# The function calls 'selection_block_greedy_screening' inside the loop but doesn't explicitly return 'selected_features' in the final list.
# I should probably modify the function to return 'selected_features' or 'selected' in the list.
# BUT, for now, I can't easily change it without potentially breaking other things (though I just fixed it).
# Wait, I SHOULD fix it to return selected features. That's critical for evaluation.

cat("Checking if selected features are returned...\n")
if (!"selected" %in% names(res) && !"selected_features" %in% names(res)) {
    cat("WARNING: Function does not return selected features. Re-running selection block on final cluster to estimate.\n")
    # Re-estimate X_tilde for final cluster (costly but necessary if not returned)
    X_tilde_final <- ISEE_bicluster(X, res$cluster)
    selected_final <- selection_block_greedy_screening(X_tilde_final, res$cluster, fdr_level = 0.1)
    res$selected <- selected_final
}

selected_indices <- which(res$selected)
cat(sprintf("Number of selected features: %d\n", length(selected_indices)))

# TP/FP
tp <- length(intersect(selected_indices, S_0))
fp <- length(setdiff(selected_indices, S_0))
fn <- length(setdiff(S_0, selected_indices))

cat(sprintf("True Positives (TP): %d / %d\n", tp, length(S_0)))
cat(sprintf("False Positives (FP): %d\n", fp))
cat(sprintf("False Negatives (FN): %d\n", fn))
cat(sprintf("Recall: %.2f\n", tp / length(S_0)))
cat(sprintf("Precision: %.2f\n", tp / length(selected_indices)))
