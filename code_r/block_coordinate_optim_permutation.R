# Permutation-Based Iterative SDP K-Means (FDR Control)
library(foreach)
library(doParallel) # Ensure parallel backend is registered if needed, though we use simple loops or standard simple parallel if implemented
library(matrixStats) # For fast colVars/rowMeans

#' Permutation-Based FDR Threshold Selector
#'
#' Estimates the FDR for a range of thresholds and selects the minimum threshold
#' that satisfies the FDR target.
#'
#' @param stats_obs Vector of observed statistics (e.g., abs_diff) from original data.
#' @param X_tilde The denoised data matrix (or original X if appropriate).
#' @param cluster_labels Current cluster assignments.
#' @param n_perms Number of permutations to estimate Null distribution.
#' @param fdr_target Target False Discovery Rate (default 0.1).
#'
#' @return A list containing:
#' \item{threshold}{The selected threshold value.}
#' \item{n_selected}{Number of features selected.}
#' \item{est_fdr}{The estimated FDR at the selected threshold.}
get_permutation_fdr_threshold <- function(stats_obs, X_tilde, cluster_labels, Omega_diag, n_perms = 20, fdr_target = 0.1) {
    # Ensure C++ function is available
    ensure_cpp_backend()

    p <- nrow(X_tilde)
    n <- length(cluster_labels)

    n1 <- sum(cluster_labels == 1)
    n2 <- sum(cluster_labels == 2)

    # 1. Sort Observed Stats (Candidates)
    sorted_stats <- sort(stats_obs, decreasing = TRUE)

    # 2. C++ Acceleration for Null Counts (SAM Style)
    # We need to map stats_obs back to "Raw Diff" scale for C++
    # OR we pass the scaling factors to C++?
    # The current C++ computes |raw_diff|.
    # Our stats_obs = |raw_diff| * scale_factor.
    # So we should pass thresholds converted to RAW scale.

    scale_factor_vec <- sqrt(n1 * n2 / n) / sqrt(pmax(Omega_diag, 1e-8))

    # But wait, scale_factor is per-feature (Omega varies).
    # If we convert thresholds to raw, we have a problem: Threshold T corresponds to different raw diffs for different features.
    # We cannot simply "unscale" the threshold globally.

    # Solution: We must compute the FULL STATISTIC in C++ (including Omega).
    # Let's verify existing C++:
    # `perm_stat = std::abs(sum1 * factor1 - factor2[i]);`
    # It does NOT include Omega.

    # Quick Fix: pass 'scale_factor_vec' to C++ and multiply inside loop.
    # I need to modify C++ again?
    # Or I can pre-scale X_tilde in R?
    # Stat = |Mean1 - Mean2| / Scale.
    #      = |Sum1/n1 - Sum2/n2| / Scale
    # If I scale X_tilde rows by (1/Scale), then new Stat = |Mean1' - Mean2'|.
    # Yes! X_tilde_scaled[i, ] = X_tilde[i, ] / Omega_term[i].
    # Then I can use the existing C++ with raw diffs on the scaled data.

    # Let's apply this trick in R to avoid changing C++ signature again.

    # statistic = abs_diff_raw * scale_factor_vec
    # We want C++ to compute: abs_diff_raw_on_input
    # If Input = X_tilde * scale_factor_vec, then
    # RawDiff(Input) = RawDiff(X) * scale_factor_vec = statistic!

    X_for_cpp <- sweep(X_tilde, 1, scale_factor_vec, "*")

    # Update factors for C++
    factor1 <- (1 / n1 + 1 / n2)
    # factor2 must be RowSums of the NEW X_for_cpp / n2
    factor2 <- rowSums(X_for_cpp) / n2

    base_indicator <- as.integer(c(rep(1, n1), rep(0, n2)))

    # Call C++
    # Counts[k] = sum(null_stats >= sorted_stats[k]) across all perms
    total_counts <- sam_perm_test_cpp(
        as.matrix(X_for_cpp), base_indicator,
        as.numeric(factor1), as.numeric(factor2),
        as.integer(n_perms), as.numeric(sorted_stats)
    )


    # 3. Grid Search (SAM)
    selected_threshold <- sqrt(5 * log(p))
    est_fdr <- 0
    best_k <- 0

    # Optimization:
    # If p is large, we don't want to loop p times checking (p * n_perms) entries.
    # But usually p ~ 400-2000. p*n_perms ~ 2e7.
    # Doing `sum(null_stats_mat >= lambda)` is fast enough (C-level scan).

    # Let's iterate.
    for (k in 1:length(sorted_stats)) {
        lambda <- sorted_stats[k]
        if (lambda <= 0) break

        # R: Discoveries in Real Data
        R <- k

        # V: Mean Discoveries in Null Data
        # sum(mat >= lambda) returns total count across all perms
        # divide by n_perms to get mean count per perm
        total_v <- sum(null_stats_mat >= lambda)
        V <- total_v / n_perms

        # FDR
        fdr_val <- V / max(R, 1)

        if (fdr_val <= fdr_target) {
            best_k <- k
            est_fdr <- fdr_val
            selected_threshold <- lambda
        }
        # In step-down, we could stop if FDR gets too high, but we search for largest k (lowest lambda)
        # matching the target.
    }

    if (best_k == 0) {
        cat("  Permutation FDR: No features met target. Reverting to Univ Threshold.\n")
        return(list(threshold = sqrt(5 * log(p)), n_selected = 0, est_fdr = NA))
    }

    return(list(threshold = selected_threshold, n_selected = best_k, est_fdr = est_fdr))
}


#' Deterministic Iterative SDP K-Means with Permutation FDR
#'
#' @param X Data matrix (p x n)
#' @param K Number of clusters
#' @param n_iter Maximum number of iterations
#' @param n_perms Number of permutations for FDR control (default 20)
#' @param fdr_target Target FDR level (default 0.1)
#' @export
block_coordinate_optim_permutation <- function(X, K, n_iter = 50, n_perms = 20, fdr_target = 0.1, stable_iter = 5, true_labels = NULL) {
    # variable initialization
    p <- nrow(X)
    n <- ncol(X)
    is_stop <- FALSE
    iternum <- 0
    rand_vec <- rep(NA, n_iter)
    consecutive_stable_count <- 0
    start_time <- Sys.time()

    # Universal threshold (Fallback)
    universal_threshold <- sqrt(5 * log(p))

    # 1. INITIALIZATION BLOCK
    cat("Running initial clustering (ESSC)...\n")
    cluster_est_now <- ESSC(X, K)

    if (!is.null(true_labels)) {
        acc_init <- get_cluster_acc(cluster_est_now, true_labels)
        cat(sprintf("Initial Clustering Accuracy: %.4f\n", acc_init))
    }

    while (!is_stop && iternum < n_iter) {
        iternum <- iternum + 1
        cat(sprintf("\n--- Iteration %d (Permutation FDR Target %.2f) ---\n", iternum, fdr_target))

        if (length(unique(cluster_est_now)) < K) {
            cat("Clusters collapsed. Stopping.\n")
            break
        }

        # --- ISEE Transformation ---
        cat("Running ISEE (Separate Slopes)...\n")
        res_isee <- ISEE_residual_lasso(X, cluster_est_now, K)
        X_tilde <- res_isee$X_tilde
        Omega_diag_hat <- res_isee$Omega_diag

        # --- Statistic Calculation ---
        # abs_diff_j
        means_mat <- matrix(0, p, K)
        for (c in 1:K) {
            means_mat[, c] <- rowMeans(X_tilde[, cluster_est_now == c, drop = FALSE])
        }
        n1 <- sum(cluster_est_now == 1)
        n2 <- sum(cluster_est_now == 2)

        # The statistic: Same as used in permutation function
        abs_diff <- abs(means_mat[, 1] - means_mat[, 2]) / sqrt(pmax(Omega_diag_hat, 1e-8)) * sqrt(n1 * n2 / n)

        # --- Permutation Thresholding ---
        cat(sprintf("Estimating Threshold via %d permutations...\n", n_perms))
        perm_res <- get_permutation_fdr_threshold(abs_diff, X_tilde, cluster_est_now, Omega_diag_hat, n_perms, fdr_target)

        current_threshold <- perm_res$threshold
        cat(sprintf("  Selected Threshold: %.4f (Univ: %.4f)\n", current_threshold, universal_threshold))
        cat(sprintf("  Features Selected: %d (Est FDR: %.4f)\n", perm_res$n_selected, perm_res$est_fdr))

        if (perm_res$n_selected == 0) {
            cat("  WARNING: No features selected. Using Top K features fallback.\n")
            s_hat <- order(abs_diff, decreasing = TRUE)[1:K]
        } else {
            s_hat <- which(abs_diff >= current_threshold)
        }

        # --- Clustering Block ---
        # Re-cluster using SDP with selected features
        cat("Running Clustering Block...\n")

        # Estimate Sigma_hat on s_hat
        Sigma_hat_small <- get_cov_small(X, cluster_est_now, s_hat)
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

        if (ari_now > 0.9999) {
            consecutive_stable_count <- consecutive_stable_count + 1
        } else {
            consecutive_stable_count <- 0
        }

        cluster_est_now <- cluster_est_new

        # Stop if stable
        if (consecutive_stable_count >= stable_iter) {
            cat("Converged: Stable for consecutive iterations.\n")
            is_stop <- TRUE
        }
    }

    total_time <- difftime(Sys.time(), start_time, units = "secs")

    return(list(
        cluster = cluster_est_now,
        s_hat = s_hat,
        iternum = iternum,
        abs_diff = abs_diff,
        final_threshold = current_threshold
    ))
}
