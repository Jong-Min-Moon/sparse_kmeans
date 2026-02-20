# Test ISEE Accuracy vs Ground Truth
library(methods)
library(Matrix)
library(glmnet)
library(foreach)
library(doParallel)
library(MASS)

# Source implementations
source("code_r/ISEE_bicluster.R")
source("code_r/get_intercept_residual_lasso.R")

# Register Parallel for testing
if (getDoParWorkers() == 1) {
    registerDoParallel(cores = 4)
}

# ---------------------------------------------------------
# 1. Setup Ground Truth Data (Matching sim02)
# ---------------------------------------------------------
set.seed(42)
n <- 200
p <- 100
K <- 2
rho <- 0.45

# Precision Matrix Omega (Tridiagonal)
Omega_true <- matrix(0, p, p)
for (i in 1:p) {
    Omega_true[i, i] <- 1
    if (i > 1) Omega_true[i, i - 1] <- rho
    if (i < p) Omega_true[i, i + 1] <- rho
}
Sigma_true <- solve(Omega_true)

# Means
mu1 <- rep(0, p)
mu1[1:5] <- 1
mu2 <- rep(0, p)
mu2[1:5] <- -1
Mu_true <- cbind(mu1, mu2)

# Generate Data
cluster_true <- c(rep(1, n/2), rep(2, n/2))
X <- matrix(0, p, n)
X[, 1:(n/2)] <- t(mvrnorm(n/2, mu1, Sigma_true))
X[, (n/2+1):n] <- t(mvrnorm(n/2, mu2, Sigma_true))

# True X_tilde: Omega * (X - Mu)
Mu_matrix <- Mu_true[, cluster_true]
X_tilde_true <- Omega_true %*% (X - Mu_matrix)

cat("\n--- Comparison vs Ground Truth (n=200, p=100) ---\n")

# 2. Run Versions
cat("Running Original ISEE...\n")
res_orig <- ISEE_bicluster(X, cluster_true)

cat("Running Stacked ISEE...\n")
res_stacked <- ISEE_bicluster_stacked(X, cluster_true)

# 3. Calculate Metrics
calc_metrics <- function(res, name) {
    # Omega Diagonal MSE
    omega_mse <- mean((res$Omega_diag_hat - diag(Omega_true))^2)
    # X_tilde MSE
    xtilde_mse <- mean((res$X_tilde - X_tilde_true)^2)
    
    # Correlation with truth
    xtilde_cor <- cor(as.numeric(res$X_tilde), as.numeric(X_tilde_true))
    
    cat(sprintf("\nResults for %s:\n", name))
    cat(sprintf("  Omega Diag MSE: %.6f\n", omega_mse))
    cat(sprintf("  X_tilde MSE:    %.6f\n", xtilde_mse))
    cat(sprintf("  X_tilde COR:    %.6f\n", xtilde_cor))
    
    return(list(omega_mse = omega_mse, xtilde_mse = xtilde_mse))
}

m_orig <- calc_metrics(res_orig, "Original (Separate Slopes)")
m_stacked <- calc_metrics(res_stacked, "Stacked (Shared Slopes)")

cat("\n--- Conclusion ---\n")
if (m_stacked$xtilde_mse < m_orig$xtilde_mse) {
    cat("Stacked version is MORE accurate for X_tilde.\n")
} else {
    cat("Original version is MORE accurate for X_tilde.\n")
}
