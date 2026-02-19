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
#' @param stable_iter Number of consecutive iterations with same labels for convergence (Deprecated in favor of ari_consecutive_stop, but kept for compatibility)
#' @param true_labels Optional ground truth for ARI tracking
#' @param ari_consecutive_stop Number of consecutive iterations with ARI=1 to stop
#' @export
block_coordinate_optim_deterministic_unknowncov <- function(X, K, n_iter = 50, stable_iter = 5, true_labels = NULL, ari_consecutive_stop = 10) {
    # variable initialization
    p <- nrow(X)
    n <- ncol(X)
    is_stop <- FALSE
    iternum <- 0
    rand_vec <- rep(NA, n_iter)
    consecutive_stable_count <- 0
    start_time <- Sys.time()
    universal_threshold <- sqrt(2 * log(p))

    # 1. INITIALIZATION BLOCK
    cat("Running initial clustering (ESSC)...\n")
    cluster_est_now <- ESSC(X, K)

    if (!is.null(true_labels)) {
        acc_init <- get_cluster_acc(cluster_est_now, true_labels)
        cat(sprintf("Initial Clustering Accuracy: %.4f\n", acc_init))
    }


    while (!is_stop && iternum < n_iter) {
        iternum <- iternum + 1
        cat(sprintf("\n--- Iteration %d (Deterministic Selection) ---\n", iternum))

        if (length(unique(cluster_est_now)) < K) {
            cat("Clusters collapsed. Stopping.\n")
            break
        }

        # --- ISEE Transformation (MATLAB Style: Separate Slopes + AIC) ---
        cat("Running ISEE (Separate Slopes, AIC Selection)...\n")

        res_isee <- ISEE_residual_lasso(X, cluster_est_now, K)
        X_tilde <- res_isee$X_tilde
        Omega_diag_hat <- res_isee$Omega_diag

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
        cat(sprintf("Stability Check (ARI vs Prev): %.4f\n", ari_now))

        if (!is.null(true_labels)) {
            acc_true <- get_cluster_acc(cluster_est_new, true_labels)
            cat(sprintf("Iteration %d Accuracy: %.4f\n", iternum, acc_true))
        }

        # Check for consecutive stability (ARI == 1)
        if (ari_now > 0.9999) { # Float tolerance for 1.0
            consecutive_stable_count <- consecutive_stable_count + 1
            cat(sprintf("Consecutive stable iterations: %d/%d\n", consecutive_stable_count, ari_consecutive_stop))
        } else {
            consecutive_stable_count <- 0
        }

        cluster_est_now <- cluster_est_new

        # Stop if stable for N consecutive iterations
        if (consecutive_stable_count >= ari_consecutive_stop) {
            cat(sprintf("Converged: ARI stable for %d consecutive iterations.\n", ari_consecutive_stop))
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
