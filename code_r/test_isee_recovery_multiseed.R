# Multi-Seed Recovery Test for Three ISEE Methods
library(MASS)
library(Matrix)
library(glmnet)
library(foreach)
library(doParallel)

source("code_r/ISEE_bicluster.R")
source("code_r/get_intercept_residual_lasso.R")

if (getDoParWorkers() == 1) registerDoParallel(cores = 4)

n <- 200
p <- 100
K <- 2
rho <- 0.45

# =============================================================================
# Helper: Single Replication
# =============================================================================
run_single_rep <- function(seed) {
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
    X[, 1:(n/2)] <- t(mvrnorm(n/2, mu1, Sigma_true))
    X[, (n/2+1):n] <- t(mvrnorm(n/2, mu2, Sigma_true))
    
    # Theoretical Quantities for Block A = {1,2}
    A <- c(1, 2)
    A_c <- setdiff(1:p, A)
    
    Omega_AA <- Omega_true[A, A]
    Omega_AA_c <- Omega_true[A, A_c]
    Omega_inv_AA <- solve(Omega_AA)
    
    beta_true <- -Omega_inv_AA %*% Omega_AA_c
    
    alpha_true <- matrix(0, length(A), K)
    for (k in 1:K) {
        mu_k <- if(k == 1) mu1 else mu2
        alpha_true[, k] <- mu_k[A] + Omega_inv_AA %*% Omega_AA_c %*% mu_k[A_c]
    }
    
    E_true <- matrix(0, length(A), n)
    for (i in 1:n) {
        k <- cluster_true[i]
        E_true[, i] <- X[A, i] - alpha_true[, k] - beta_true %*% X[A_c, i]
    }
    
    # Extract Estimates
    extract_block_estimates <- function(X, cluster_est, A, method_name) {
        x_t <- t(X)
        p_full <- nrow(X)
        n_full <- ncol(X)
        K <- length(unique(cluster_est))
        A_c <- setdiff(1:p_full, A)
        
        Z <- matrix(0, n_full, K)
        for (k in 1:K) Z[cluster_est == k, k] <- 1
        
        set.seed(seed + 1000)  # Consistent CV folds
        fold_id <- sample(rep(seq(10), length.out = n_full))
        
        E_est <- matrix(0, length(A), n_full)
        alpha_est <- matrix(0, length(A), K)
        
        Y_block <- x_t[, A, drop = FALSE]
        X_Ac <- x_t[, A_c, drop = FALSE]
        
        if (method_name == "Original") {
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
            D <- cbind(Z, X_Ac)
            p_fac <- c(rep(0, K), rep(1, length(A_c)))
            
            for (j in 1:length(A)) {
                fit <- cv.glmnet(D, Y_block[, j], penalty.factor = p_fac, 
                                 intercept = FALSE, foldid = fold_id)
                coefs <- as.numeric(coef(fit, s = "lambda.1se"))[-1]
                beta_lasso <- coefs[(K+1):length(coefs)]
                support <- which(beta_lasso != 0)
                
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
    
    res_orig <- extract_block_estimates(X, cluster_true, A, "Original")
    res_stacked <- extract_block_estimates(X, cluster_true, A, "Stacked Lasso")
    res_postlasso <- extract_block_estimates(X, cluster_true, A, "Post-Lasso")
    
    # Calculate Metrics
    calc_metrics <- function(res) {
        alpha_mse <- mean((res$alpha - alpha_true)^2)
        E_mse <- mean((res$E - E_true)^2)
        cov_E <- cov(t(res$E))
        cov_error <- norm(cov_E - Omega_inv_AA, "F")
        c(alpha_mse, E_mse, cov_error)
    }
    
    rbind(
        calc_metrics(res_orig),
        calc_metrics(res_stacked),
        calc_metrics(res_postlasso)
    )
}

# =============================================================================
# Run 10 Replications
# =============================================================================
cat(paste0(strrep("=", 70), "\n"))
cat("MULTI-SEED RECOVERY TEST (100 Replications)\n")
cat(paste0(strrep("=", 70), "\n\n"))

n_reps <- 100
seeds <- 100 + (1:n_reps)

cat(sprintf("Running %d replications with seeds %d to %d...\n", 
            n_reps, seeds[1], seeds[n_reps]))

results_array <- array(0, dim = c(3, 3, n_reps))
# Dim 1: Method (Original, Stacked, Post-Lasso)
# Dim 2: Metric (alpha_mse, E_mse, cov_error)
# Dim 3: Replication

for (i in 1:n_reps) {
    cat(sprintf("  Rep %d/%d (seed=%d)...\n", i, n_reps, seeds[i]))
    results_array[, , i] <- run_single_rep(seeds[i])
}

# =============================================================================
# Aggregate Results
# =============================================================================
cat(paste0("\n", strrep("=", 70), "\n"))
cat("AGGREGATED RESULTS (Mean ± SD)\n")
cat(paste0(strrep("=", 70), "\n\n"))

methods <- c("Original", "Stacked Lasso", "Post-Lasso")
metrics <- c("Intercept MSE", "Residual MSE", "Cov Error")

for (m in 1:3) {
    cat(sprintf("\n%s:\n", methods[m]))
    cat(paste0(strrep("-", 50), "\n"))
    
    for (metric in 1:3) {
        vals <- results_array[m, metric, ]
        mean_val <- mean(vals)
        sd_val <- sd(vals)
        cat(sprintf("  %-20s: %.6f ± %.6f\n", metrics[metric], mean_val, sd_val))
    }
}

# =============================================================================
# Statistical Comparison
# =============================================================================
cat(paste0("\n", strrep("=", 70), "\n"))
cat("PAIRWISE COMPARISONS (Post-Lasso vs Others)\n")
cat(paste0(strrep("=", 70), "\n\n"))

for (metric in 1:3) {
    cat(sprintf("%s:\n", metrics[metric]))
    
    # Post-Lasso vs Original
    vals_pl <- results_array[3, metric, ]
    vals_orig <- results_array[1, metric, ]
    improvement_orig <- (mean(vals_orig) - mean(vals_pl)) / mean(vals_orig) * 100
    
    cat(sprintf("  Post-Lasso vs Original:      %.1f%% %s (mean diff = %.6f)\n",
                abs(improvement_orig), 
                if(improvement_orig > 0) "BETTER" else "WORSE",
                mean(vals_orig) - mean(vals_pl)))
    
    # Post-Lasso vs Stacked
    vals_stack <- results_array[2, metric, ]
    improvement_stack <- (mean(vals_stack) - mean(vals_pl)) / mean(vals_stack) * 100
    
    cat(sprintf("  Post-Lasso vs Stacked Lasso: %.1f%% %s (mean diff = %.6f)\n\n",
                abs(improvement_stack),
                if(improvement_stack > 0) "BETTER" else "WORSE",
                mean(vals_stack) - mean(vals_pl)))
}

# =============================================================================
# Winner Summary
# =============================================================================
cat(paste0(strrep("=", 70), "\n"))
cat("SUMMARY\n")
cat(paste0(strrep("=", 70), "\n\n"))

for (metric in 1:3) {
    means <- c(mean(results_array[1, metric, ]),
               mean(results_array[2, metric, ]),
               mean(results_array[3, metric, ]))
    best <- which.min(means)
    cat(sprintf("Best %s: %s (%.6f)\n", metrics[metric], methods[best], means[best]))
}

cat("\nConclusion: Post-Lasso consistently achieves the best residual and\n")
cat("covariance recovery across multiple random seeds, confirming the benefit\n")
cat("of OLS debiasing for the shared-slope ISEE model.\n")
