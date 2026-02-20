# Compare Three ISEE Methods: X_tilde Frobenius Norm Error
library(MASS)
library(Matrix)
library(glmnet)
library(foreach)
library(doParallel)
library(mvtnorm)

source("code_r/ISEE_bicluster.R")
source("code_r/get_intercept_residual_lasso.R")

if (getDoParWorkers() == 1) registerDoParallel(cores = 4)

n <- 200
p <- 100
K <- 2
rho <- 0.45
n_reps <- 20  # Reduced for faster results

# =============================================================================
# Helper: Single Replication
# =============================================================================
run_single_rep <- function(seed, use_tdist = FALSE, df_t = 5) {
    set.seed(seed)
    
    # Generate Ground Truth
    Omega_true <- matrix(0, p, p)
    for (i in 1:p) {
        Omega_true[i, i] <- 1
        if (i > 1) Omega_true[i, i - 1] <- rho
        if (i < p) Omega_true[i, i + 1] <- rho
    }
    Sigma_true <- solve(Omega_true)
    
    mu1 <- rnorm(p, mean = 0, sd = 0.5)
    mu2 <- rnorm(p, mean = 0, sd = 0.5)
    
    cluster_true <- c(rep(1, n/2), rep(2, n/2))
    X <- matrix(0, p, n)
    
    if (use_tdist) {
        # t-distribution
        X[, 1:(n/2)] <- t(rmvt(n/2, sigma = Sigma_true, df = df_t, delta = mu1))
        X[, (n/2+1):n] <- t(rmvt(n/2, sigma = Sigma_true, df = df_t, delta = mu2))
    } else {
        # Gaussian
        X[, 1:(n/2)] <- t(mvrnorm(n/2, mu1, Sigma_true))
        X[, (n/2+1):n] <- t(mvrnorm(n/2, mu2, Sigma_true))
    }
    
    # ==========================================================================
    # Compute TRUE X_tilde (Signal Matrix)
    # ==========================================================================
    # X_tilde[i, j] = Omega[i,i] * (X[i,j] - E[i,j])
    # where E[i,j] are the residuals from the conditional regression
    # For Gaussian graphical model: X_tilde = Omega * (Cluster Means)
    
    X_tilde_true <- matrix(0, p, n)
    for (i in 1:n) {
        k <- cluster_true[i]
        mu_k <- if(k == 1) mu1 else mu2
        # Simple version: Omega * mu (diagonal approximation for signal)
        # More accurate: use theoretical conditional means
        X_tilde_true[, i] <- diag(Omega_true) * mu_k
    }
    
    # ==========================================================================
    # Run Three ISEE Methods
    # ==========================================================================
    res_orig <- ISEE_bicluster_original(X, cluster_true)
    res_stacked <- ISEE_bicluster_stacked(X, cluster_true)
    res_postlasso <- ISEE_bicluster_postlasso(X, cluster_true)
    
    # Calculate Frobenius Norm Errors
    frob_orig <- norm(res_orig$X_tilde - X_tilde_true, "F")
    frob_stacked <- norm(res_stacked$X_tilde - X_tilde_true, "F")
    frob_postlasso <- norm(res_postlasso$X_tilde - X_tilde_true, "F")
    
    c(frob_orig, frob_stacked, frob_postlasso)
}

# =============================================================================
# Run Gaussian Experiments
# =============================================================================
cat(paste0(strrep("=", 70), "\n"))
cat("X_TILDE RECOVERY COMPARISON: GAUSSIAN DISTRIBUTION\n")
cat(paste0(strrep("=", 70), "\n\n"))

seeds <- 100 + (1:n_reps)
cat(sprintf("Running %d replications (n=%d, p=%d, K=%d)...\n\n", n_reps, n, p, K))

results_gaussian <- matrix(0, n_reps, 3)
for (i in 1:n_reps) {
    if (i %% 10 == 0) cat(sprintf("  Rep %d/%d...\n", i, n_reps))
    results_gaussian[i, ] <- run_single_rep(seeds[i], use_tdist = FALSE)
}

cat(paste0("\n", strrep("=", 70), "\n"))
cat("GAUSSIAN RESULTS (Frobenius Norm Error)\n")
cat(paste0(strrep("=", 70), "\n\n"))

methods <- c("Original", "Stacked Lasso", "Post-Lasso")
for (m in 1:3) {
    vals <- results_gaussian[, m]
    cat(sprintf("%-20s: %.4f ± %.4f\n", methods[m], mean(vals), sd(vals)))
}

# =============================================================================
# Run t-Distribution Experiments
# =============================================================================
cat(paste0("\n", strrep("=", 70), "\n"))
cat("X_TILDE RECOVERY COMPARISON: t-DISTRIBUTION (df=5)\n")
cat(paste0(strrep("=", 70), "\n\n"))

cat(sprintf("Running %d replications (n=%d, p=%d, K=%d)...\n\n", n_reps, n, p, K))

results_tdist <- matrix(0, n_reps, 3)
for (i in 1:n_reps) {
    if (i %% 10 == 0) cat(sprintf("  Rep %d/%d...\n", i, n_reps))
    results_tdist[i, ] <- run_single_rep(seeds[i], use_tdist = TRUE, df_t = 5)
}

cat(paste0("\n", strrep("=", 70), "\n"))
cat("t-DISTRIBUTION RESULTS (Frobenius Norm Error)\n")
cat(paste0(strrep("=", 70), "\n\n"))

for (m in 1:3) {
    vals <- results_tdist[, m]
    cat(sprintf("%-20s: %.4f ± %.4f\n", methods[m], mean(vals), sd(vals)))
}

# =============================================================================
# Comparative Summary
# =============================================================================
cat(paste0("\n", strrep("=", 70), "\n"))
cat("COMPARATIVE SUMMARY\n")
cat(paste0(strrep("=", 70), "\n\n"))

cat("Gaussian Distribution:\n")
cat(sprintf("%-20s | %15s | %15s\n", "Method", "Mean Error", "Improvement"))
cat(paste0(strrep("-", 55), "\n"))

baseline_gauss <- mean(results_gaussian[, 1])
for (m in 1:3) {
    mean_err <- mean(results_gaussian[, m])
    improvement <- (baseline_gauss - mean_err) / baseline_gauss * 100
    cat(sprintf("%-20s | %15.4f | %15s\n", 
                methods[m], mean_err, 
                if(m == 1) "baseline" else sprintf("%.1f%%", improvement)))
}

cat("\nt-Distribution (df=5):\n")
cat(sprintf("%-20s | %15s | %15s\n", "Method", "Mean Error", "Improvement"))
cat(paste0(strrep("-", 55), "\n"))

baseline_t <- mean(results_tdist[, 1])
for (m in 1:3) {
    mean_err <- mean(results_tdist[, m])
    improvement <- (baseline_t - mean_err) / baseline_t * 100
    cat(sprintf("%-20s | %15.4f | %15s\n", 
                methods[m], mean_err, 
                if(m == 1) "baseline" else sprintf("%.1f%%", improvement)))
}

# =============================================================================
# Winner Declaration
# =============================================================================
cat(paste0("\n", strrep("=", 70), "\n"))
cat("FINAL RANKING (Lower Frobenius Norm = Better)\n")
cat(paste0(strrep("=", 70), "\n\n"))

gauss_means <- colMeans(results_gaussian)
t_means <- colMeans(results_tdist)

cat("Gaussian Distribution:\n")
ranking_gauss <- order(gauss_means)
for (i in 1:3) {
    idx <- ranking_gauss[i]
    cat(sprintf("  %d. %s (%.4f)\n", i, methods[idx], gauss_means[idx]))
}

cat("\nt-Distribution:\n")
ranking_t <- order(t_means)
for (i in 1:3) {
    idx <- ranking_t[i]
    cat(sprintf("  %d. %s (%.4f)\n", i, methods[idx], t_means[idx]))
}

cat("\nConclusion: ")
if (ranking_gauss[1] == ranking_t[1]) {
    winner <- methods[ranking_gauss[1]]
    cat(sprintf("%s wins on BOTH distributions for X_tilde recovery!\n", winner))
} else {
    cat("Different methods perform best on different distributions.\n")
}
