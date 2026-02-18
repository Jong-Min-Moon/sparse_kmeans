# Test Intercept and Residual Recovery for Three ISEE Methods
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
p <- 50
K <- 2
rho <- 0.45

# =============================================================================
# 1. Generate Ground Truth
# =============================================================================
cat("Generating ground truth data...\n")

# Precision Matrix (Tridiagonal)
Omega_true <- matrix(0, p, p)
for (i in 1:p) {
    Omega_true[i, i] <- 1
    if (i > 1) Omega_true[i, i - 1] <- rho
    if (i < p) Omega_true[i, i + 1] <- rho
}
Sigma_true <- solve(Omega_true)

# Cluster-specific means
mu1 <- rnorm(p, mean = 0, sd = 0.5)
mu2 <- rnorm(p, mean = 0, sd = 0.5)

# Generate data
cluster_true <- c(rep(1, n/2), rep(2, n/2))
X <- matrix(0, p, n)
X[, 1:(n/2)] <- t(mvrnorm(n/2, mu1, Sigma_true))
X[, (n/2+1):n] <- t(mvrnorm(n/2, mu2, Sigma_true))

# =============================================================================
# 2. Compute Theoretical Quantities for Block A = {1,2}
# =============================================================================
cat("\nComputing theoretical intercepts and residuals for block A = {1,2}...\n")

A <- c(1, 2)
A_c <- setdiff(1:p, A)

Omega_AA <- Omega_true[A, A]
Omega_AA_c <- Omega_true[A, A_c]
Omega_inv_AA <- solve(Omega_AA)

# Theoretical slope (SAME for all clusters)
beta_true <- -Omega_inv_AA %*% Omega_AA_c

# Theoretical intercepts (cluster-specific)
alpha_true <- matrix(0, length(A), K)
for (k in 1:K) {
    mu_k <- if(k == 1) mu1 else mu2
    alpha_true[, k] <- mu_k[A] + Omega_inv_AA %*% Omega_AA_c %*% mu_k[A_c]
}

# Theoretical residuals
E_true <- matrix(0, length(A), n)
for (i in 1:n) {
    k <- cluster_true[i]
    E_true[, i] <- X[A, i] - alpha_true[, k] - beta_true %*% X[A_c, i]
}

cat(sprintf("  True intercept (cluster 1): [%.4f, %.4f]\n", alpha_true[1,1], alpha_true[2,1]))
cat(sprintf("  True intercept (cluster 2): [%.4f, %.4f]\n", alpha_true[1,2], alpha_true[2,2]))

# =============================================================================
# 3. Extract Block Estimates from All Three Methods
# =============================================================================

extract_block_estimates <- function(X, cluster_est, A, method_name, method_func) {
    x_t <- t(X)
    p_full <- nrow(X)
    n_full <- ncol(X)
    K <- length(unique(cluster_est))
    A_c <- setdiff(1:p_full, A)
    
    Z <- matrix(0, n_full, K)
    for (k in 1:K) Z[cluster_est == k, k] <- 1
    
    set.seed(42)
    fold_id <- sample(rep(seq(10), length.out = n_full))
    
    E_est <- matrix(0, length(A), n_full)
    alpha_est <- matrix(0, length(A), K)
    
    Y_block <- x_t[, A, drop = FALSE]
    X_Ac <- x_t[, A_c, drop = FALSE]
    
    if (method_name == "Original") {
        # Per-cluster regression
        for (k in 1:K) {
            mask <- (cluster_est == k)
            X_Ac_k <- X_Ac[mask, , drop = FALSE]
            
            for (j in 1:length(A)) {
                y_k <- Y_block[mask, j]
                fit <- cv.glmnet(X_Ac_k, y_k, foldid = fold_id[mask])
                coefs <- as.numeric(coef(fit, s = "lambda.1se"))
                alpha_est[j, k] <- coefs[1]
                beta_est <- coefs[-1]
                E_est[j, mask] <- y_k - coefs[1] - X_Ac_k %*% beta_est
            }
        }
    } else if (method_name == "Stacked Lasso") {
        # Stacked Lasso
        D <- cbind(Z, X_Ac)
        p_fac <- c(rep(0, K), rep(1, length(A_c)))
        
        for (j in 1:length(A)) {
            fit <- cv.glmnet(D, Y_block[, j], penalty.factor = p_fac, 
                             intercept = FALSE, foldid = fold_id)
            coefs <- as.numeric(coef(fit, s = "lambda.1se"))[-1]
            alpha_est[j, ] <- coefs[1:K]
            beta_est <- coefs[(K+1):length(coefs)]
            E_est[j, ] <- Y_block[, j] - D %*% coefs
        }
    } else if (method_name == "Post-Lasso") {
        # Stage 1: Lasso
        D <- cbind(Z, X_Ac)
        p_fac <- c(rep(0, K), rep(1, length(A_c)))
        
        for (j in 1:length(A)) {
            fit <- cv.glmnet(D, Y_block[, j], penalty.factor = p_fac, 
                             intercept = FALSE, foldid = fold_id)
            coefs <- as.numeric(coef(fit, s = "lambda.1se"))[-1]
            beta_lasso <- coefs[(K+1):length(coefs)]
            support <- which(beta_lasso != 0)
            
            # Stage 2: OLS Refit
            if (length(support) == 0) {
                for(k in 1:K) {
                    mask <- (cluster_est == k)
                    if(sum(mask) > 0) alpha_est[j, k] <- mean(Y_block[mask, j])
                }
                E_est[j, ] <- Y_block[, j] - Z %*% alpha_est[j, ]
            } else {
                D_refit <- cbind(Z, X_Ac[, support, drop = FALSE])
                fit_ols <- lm.fit(D_refit, Y_block[, j])
                coefs_ols <- fit_ols$coefficients
                alpha_est[j, ] <- coefs_ols[1:K]
                beta_ols <- coefs_ols[(K+1):(K+length(support))]
                E_est[j, ] <- Y_block[, j] - (Z %*% alpha_est[j, ] + X_Ac[, support, drop = FALSE] %*% beta_ols)
            }
        }
    }
    
    list(alpha = alpha_est, E = E_est)
}

cat(paste0("\n", strrep("=", 70), "\n"))
cat("RUNNING THREE ISEE METHODS\n")
cat(paste0(strrep("=", 70), "\n\n"))

cat("1. Original (Separate Slopes)...\n")
res_orig <- extract_block_estimates(X, cluster_true, A, "Original", NULL)

cat("2. Stacked Lasso (Shared Slopes, Lasso)...\n")
res_stacked <- extract_block_estimates(X, cluster_true, A, "Stacked Lasso", NULL)

cat("3. Post-Lasso (Shared Slopes, OLS)...\n")
res_postlasso <- extract_block_estimates(X, cluster_true, A, "Post-Lasso", NULL)

# =============================================================================
# 4. Compare Recovery
# =============================================================================
cat(paste0("\n", strrep("=", 70), "\n"))
cat("RECOVERY COMPARISON (Block A = {1, 2})\n")
cat(paste0(strrep("=", 70), "\n\n"))

calc_recovery_metrics <- function(res, name) {
    alpha_mse <- mean((res$alpha - alpha_true)^2)
    E_mse <- mean((res$E - E_true)^2)
    cov_E <- cov(t(res$E))
    cov_error <- norm(cov_E - Omega_inv_AA, "F")
    
    list(name = name, alpha_mse = alpha_mse, E_mse = E_mse, cov_error = cov_error)
}

m_orig <- calc_recovery_metrics(res_orig, "Original")
m_stacked <- calc_recovery_metrics(res_stacked, "Stacked Lasso")
m_postlasso <- calc_recovery_metrics(res_postlasso, "Post-Lasso")

cat(sprintf("%-20s | %15s | %15s | %15s\n", 
            "Method", "Intercept MSE", "Residual MSE", "Cov Error"))
cat(paste0(strrep("-", 70), "\n"))

print_row <- function(m) {
    cat(sprintf("%-20s | %15.6f | %15.6f | %15.6f\n",
                m$name, m$alpha_mse, m$E_mse, m$cov_error))
}

print_row(m_orig)
print_row(m_stacked)
print_row(m_postlasso)

# =============================================================================
# 5. Verdict
# =============================================================================
cat(paste0("\n", strrep("=", 70), "\n"))
cat("VERDICT\n")
cat(paste0(strrep("=", 70), "\n\n"))

best_alpha <- which.min(c(m_orig$alpha_mse, m_stacked$alpha_mse, m_postlasso$alpha_mse))
best_E <- which.min(c(m_orig$E_mse, m_stacked$E_mse, m_postlasso$E_mse))
methods <- c("Original", "Stacked Lasso", "Post-Lasso")

cat(sprintf("Best Intercept Recovery: %s (MSE = %.6f)\n", 
            methods[best_alpha], c(m_orig$alpha_mse, m_stacked$alpha_mse, m_postlasso$alpha_mse)[best_alpha]))
cat(sprintf("Best Residual Recovery:  %s (MSE = %.6f)\n", 
            methods[best_E], c(m_orig$E_mse, m_stacked$E_mse, m_postlasso$E_mse)[best_E]))

cat("\nKey Insight: Post-Lasso OLS refit removes Lasso bias, improving recovery\n")
cat("of theoretical intercepts and residuals for the shared-slope model.\n")
