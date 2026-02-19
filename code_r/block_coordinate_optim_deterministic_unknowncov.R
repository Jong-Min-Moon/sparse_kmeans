# Deterministic Iterative SDP K-Means with Unknown Covariance (MATLAB Variant)
library(foreach)

#' Deterministic Iterative SDP K-Means with Unknown Covariance
#'
#' This implementation follows the MATLAB logic:
#' 1. Initial clustering using MATLAB-style spectral clustering.
#' 2. ISEE transformation with separate regressions per cluster.
#' 3. Feature selection using a universal threshold sqrt(2*log(p)).
#' 4. Iterative updates until convergence.
#'
#' @param X Data matrix (p x n)
#' @param K Number of clusters
#' @param n_iter Maximum number of iterations
#' @param stable_iter Number of consecutive iterations with same labels for convergence
#' @param true_labels Optional ground truth for ARI tracking
#' @export
block_coordinate_optim_deterministic_unknowncov <- function(X, K, n_iter = 50, stable_iter = 5, true_labels = NULL) {
    p <- nrow(X)
    n <- ncol(X)
    start_time <- Sys.time()

    # 1. INITIALIZATION BLOCK
    cat("Running initial clustering (MATLAB style cluster_spectral)...\n")
    cluster_est_now <- cluster_spectral_matlab(X, K)

    if (!is.null(true_labels)) {
        ari_init <- mclust::adjustedRandIndex(cluster_est_now, true_labels)
        cat(sprintf("Initial Clustering Accuracy (ARI): %.4f\n", ari_init))
    }

    is_stop <- FALSE
    iternum <- 0
    rand_vec <- rep(NA, n_iter)
    consecutive_stable_count <- 0

    universal_threshold <- sqrt(2 * log(p))

    while (!is_stop && iternum < n_iter) {
        iternum <- iternum + 1
        cat(sprintf("\n--- Iteration %d (Deterministic Selection) ---\n", iternum))

        if (length(unique(cluster_est_now)) < K) {
            cat("Clusters collapsed. Stopping.\n")
            break
        }

        # --- ISEE Transformation (MATLAB Style: Separate Slopes + AIC) ---
        cat("Running ISEE (Separate Slopes, AIC Selection)...\n")

        x_t <- t(X)
        X_tilde <- matrix(0, nrow = p, ncol = n)
        Omega_diag_hat <- numeric(p)
        n_regression <- floor(p / 2)

        results_list <- foreach::foreach(i = 1:n_regression, .packages = c("glmnet"), .export = c("get_intercept_residual_lasso_aic")) %dopar% {
            rows_idx <- c(2 * i - 1, 2 * i)
            predictor_all <- x_t[, -rows_idx, drop = FALSE]
            E_Al <- matrix(0, 2, n)
            alpha_Al <- matrix(0, 2, K)

            for (c in 1:K) {
                cluster_mask <- (cluster_est_now == c)
                if (sum(cluster_mask) < 2) next
                predictor_cluster <- predictor_all[cluster_mask, , drop = FALSE]
                for (j in 1:2) {
                    row_id <- rows_idx[j]
                    response_cluster <- x_t[cluster_mask, row_id]
                    res <- get_intercept_residual_lasso_aic(response_cluster, predictor_cluster)
                    E_Al[j, cluster_mask] <- res$residual
                    alpha_Al[j, c] <- res$intercept
                }
            }

            # Robustness: variance floor
            scatter_mat <- tcrossprod(E_Al)
            Omega_hat_Al <- tryCatch(
                {
                    solve(scatter_mat + diag(1e-4, 2)) * n
                },
                error = function(e) {
                    diag(1 / pmax(diag(scatter_mat), 1e-4)) * n
                }
            )

            X_tilde_local <- Omega_hat_Al %*% (alpha_Al[, cluster_est_now] + E_Al)
            list(rows_idx = rows_idx, X_tilde_local = X_tilde_local, diag_local = diag(Omega_hat_Al))
        }

        for (res in results_list) {
            X_tilde[res$rows_idx, ] <- res$X_tilde_local
            Omega_diag_hat[res$rows_idx] <- res$diag_local
        }

        # --- Feature Selection ---
        # abs_diff_j = |mean(X_tilde[j, C1]) - mean(X_tilde[j, C2])| / sqrt(Omega_diag_hat[j])
        abs_diff <- numeric(p)
        means_mat <- matrix(0, p, K)
        for (c in 1:K) {
            means_mat[, c] <- rowMeans(X_tilde[, cluster_est_now == c, drop = FALSE])
        }
        # For K=2, it is simply the absolute difference
        n1 <- sum(cluster_est_now == 1)
        n2 <- sum(cluster_est_now == 2)
        abs_diff <- abs(means_mat[, 1] - means_mat[, 2]) / sqrt(pmax(Omega_diag_hat, 1e-8)) * sqrt(n1 * n2 / n)

        s_hat <- which(abs_diff > universal_threshold)
        cat(sprintf("Selected %d features using universal threshold (%.3f)\n", length(s_hat), universal_threshold))

        # Debug: Top 10
        top_idx <- order(abs_diff, decreasing = TRUE)[1:10]
        cat(sprintf("Top 10 candidates by abs_diff: %s \n", paste(top_idx, collapse = ", ")))
        cat(sprintf("abs_diff values: %s \n", paste(round(abs_diff[top_idx], 2), collapse = ", ")))

        # --- Clustering Block ---
        if (length(s_hat) < K) {
            cat("Too few features selected. Using top K features as fallback.\n")
            s_hat <- order(abs_diff, decreasing = TRUE)[1:K]
        }

        # Re-cluster using SDP
        cat("Running Clustering Block (with local covariance estimation)...\n")
        # Estimate Sigma_hat on s_hat
        Sigma_hat_small <- get_cov_small(X, cluster_est_now, s_hat)

        # We pass a full p x p covariance matrix (or a list/mapping if we want efficiency)
        # But run_clustering_block_knowncov expects a matrix whose size matches X_tilde (p x p)
        # Actually, let's update run_clustering_block_knowncov to accept the small Sigma directly
        # OR just build the sparse p x p version.

        Sigma_full <- diag(1, p)
        Sigma_full[s_hat, s_hat] <- Sigma_hat_small

        res_cluster <- run_clustering_block_knowncov(X_tilde, s_hat, K, cluster_est_now, covariance = Sigma_full)
        cluster_est_new <- res_cluster$cluster
        # --- Evaluation & Convergence ---
        ari_now <- mclust::adjustedRandIndex(cluster_est_new, cluster_est_now)
        if (!is.null(true_labels)) {
            ari_true <- mclust::adjustedRandIndex(cluster_est_new, true_labels)
            cat(sprintf("Iteration %d Accuracy (ARI vs True): %.4f\n", iternum, ari_true))
        }

        if (ari_now > 0.999) {
            consecutive_stable_count <- consecutive_stable_count + 1
        } else {
            consecutive_stable_count <- 0
        }

        cluster_est_now <- cluster_est_new

        if (consecutive_stable_count >= stable_iter) {
            cat(sprintf("Converged: Stable for %d iterations.\n", stable_iter))
            is_stop <- TRUE
        }
    }

    total_time <- difftime(Sys.time(), start_time, units = "secs")
    cat(sprintf("Total time: %.2f seconds\n", total_time))

    return(list(
        cluster = cluster_est_now,
        s_hat = s_hat,
        iternum = iternum,
        abs_diff = abs_diff
    ))
}
