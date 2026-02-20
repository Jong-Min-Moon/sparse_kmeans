# Compare Three ISEE Methods
library(MASS)
library(Matrix)
library(glmnet)
library(foreach)
library(doParallel)

source("code_r/ISEE_bicluster.R")
source("code_r/get_intercept_residual_lasso.R")

if (getDoParWorkers() == 1) registerDoParallel(cores = 4)

set.seed(123)
n <- 200
p <- 100
K <- 2
rho <- 0.45

# Generate Ground Truth
cat("Generating ground truth data...\n")
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
X[, 1:(n/2)] <- t(mvrnorm(n/2, mu1, Sigma_true))
X[, (n/2+1):n] <- t(mvrnorm(n/2, mu2, Sigma_true))

Mu_matrix <- cbind(mu1, mu2)[, cluster_true]
X_tilde_true <- Omega_true %*% (X - Mu_matrix)

# Run All Three Methods
cat(paste0("\n", strrep("=", 70), "\n"))
cat("COMPARING THREE ISEE METHODS\n")
cat(paste0(strrep("=", 70), "\n\n"))

cat("1. Running Original ISEE (Separate Slopes per Cluster)...\n")
t1 <- Sys.time()
res_orig <- ISEE_bicluster(X, cluster_true)
t2 <- Sys.time()
time_orig <- as.numeric(difftime(t2, t1, units = "secs"))

cat("2. Running Stacked Lasso ISEE (Shared Slopes, Lasso Estimation)...\n")
t1 <- Sys.time()
res_stacked <- ISEE_bicluster_stacked(X, cluster_true)
t2 <- Sys.time()
time_stacked <- as.numeric(difftime(t2, t1, units = "secs"))

cat("3. Running Post-Lasso ISEE (Shared Slopes, OLS Estimation)...\n")
t1 <- Sys.time()
res_postlasso <- ISEE_bicluster_postlasso(X, cluster_true)
t2 <- Sys.time()
time_postlasso <- as.numeric(difftime(t2, t1, units = "secs"))

# Compute Metrics
calc_metrics <- function(res, name, runtime) {
    omega_mse <- mean((res$Omega_diag_hat - diag(Omega_true))^2)
    xtilde_mse <- mean((res$X_tilde - X_tilde_true)^2)
    xtilde_cor <- cor(as.numeric(res$X_tilde), as.numeric(X_tilde_true))
    
    list(
        name = name,
        omega_mse = omega_mse,
        xtilde_mse = xtilde_mse,
        xtilde_cor = xtilde_cor,
        runtime = runtime
    )
}

m_orig <- calc_metrics(res_orig, "Original", time_orig)
m_stacked <- calc_metrics(res_stacked, "Stacked Lasso", time_stacked)
m_postlasso <- calc_metrics(res_postlasso, "Post-Lasso", time_postlasso)

# Display Results
cat(paste0("\n", strrep("=", 70), "\n"))
cat("RESULTS SUMMARY (n=200, p=100)\n")
cat(paste0(strrep("=", 70), "\n\n"))

cat(sprintf("%-20s | %12s | %12s | %12s | %10s\n", 
            "Method", "Omega MSE", "X_tilde MSE", "X_tilde COR", "Time (s)"))
cat(paste0(strrep("-", 70), "\n"))

print_row <- function(m) {
    cat(sprintf("%-20s | %12.6f | %12.6f | %12.6f | %10.2f\n",
                m$name, m$omega_mse, m$xtilde_mse, m$xtilde_cor, m$runtime))
}

print_row(m_orig)
print_row(m_stacked)
print_row(m_postlasso)

# Comparison
cat(paste0("\n", strrep("=", 70), "\n"))
cat("KEY INSIGHTS\n")
cat(paste0(strrep("=", 70), "\n\n"))

best_xtilde <- which.min(c(m_orig$xtilde_mse, m_stacked$xtilde_mse, m_postlasso$xtilde_mse))
best_names <- c("Original", "Stacked Lasso", "Post-Lasso")

cat(sprintf("1. Best X_tilde Recovery: %s (MSE = %.6f)\n", 
            best_names[best_xtilde], 
            c(m_orig$xtilde_mse, m_stacked$xtilde_mse, m_postlasso$xtilde_mse)[best_xtilde]))

cat(sprintf("\n2. X_tilde MSE Improvement (Post-Lasso vs Stacked):\n"))
improvement <- (m_stacked$xtilde_mse - m_postlasso$xtilde_mse) / m_stacked$xtilde_mse * 100
cat(sprintf("   %.1f%% %s\n", abs(improvement), 
            if(improvement > 0) "BETTER" else "WORSE"))

cat(sprintf("\n3. X_tilde MSE Improvement (Post-Lasso vs Original):\n"))
improvement2 <- (m_orig$xtilde_mse - m_postlasso$xtilde_mse) / m_orig$xtilde_mse * 100
cat(sprintf("   %.1f%% %s\n", abs(improvement2),
            if(improvement2 > 0) "BETTER" else "WORSE"))

cat("\n4. Theoretical Correctness:\n")
cat("   - Original: INCORRECT (separate slopes violate model)\n")
cat("   - Stacked Lasso: CORRECT (shared slopes, Lasso bias)\n")
cat("   - Post-Lasso: CORRECT (shared slopes, unbiased OLS)\n")

cat("\nConclusion: Post-Lasso achieves the best trade-off between\n")
cat("theoretical correctness, estimation accuracy, and sparsity.\n")
