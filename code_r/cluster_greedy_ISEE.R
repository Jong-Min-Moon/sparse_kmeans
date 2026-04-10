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
# Load C++ Backend (Shared Library)
# Should be compiled via R CMD SHLIB selection_utils.cpp
load_cpp_backend <- function() {
    dll_name <- "selection_utils"
    ext <- .Platform$dynlib.ext

    # Paths to check
    paths <- c(
        file.path(".", paste0(dll_name, ext)),
        file.path("code_r", paste0(dll_name, ext)),
        file.path("../../code_r", paste0(dll_name, ext)),
        file.path("../code_r", paste0(dll_name, ext))
    )

    loaded <- FALSE
    for (p in paths) {
        if (file.exists(p)) {
            dyn.load(p)
            loaded <- TRUE
            break
        }
    }

    if (!loaded) {
        warning("Could not find selection_utils", ext, ". Ensure it is compiled via R CMD SHLIB.")
    }
}

# Run loader
load_cpp_backend()

# Wrapper functions for .Call
get_signed_perm_stats_cpp <- function(X, indicator, factor1, factor2, n_perms) {
    .Call("get_signed_perm_stats_wrapper", X, indicator, factor1, factor2, n_perms)
}

count_matrix_exceedances_cpp <- function(perm_mat, thresholds) {
    .Call("count_matrix_exceedances_wrapper", perm_mat, thresholds)
}

#' SAM-Style Permutation FDR (Delta Thresholding)
#'
#' w_j = |d_j - mean(d_j^perm)|
get_permutation_fdr_threshold <- function(stats_obs, X_tilde, cluster_labels, Omega_diag, n_perms = 20, fdr_target = 0.1) {
    # C++ backend should be loaded globally

    p <- nrow(X_tilde)
    n <- length(cluster_labels)
    n1 <- sum(cluster_labels == 1)
    n2 <- sum(cluster_labels == 2)

    # 1. Prepare Scaling Factors
    scale_factor_vec <- sqrt(n1 * n2 / n) / sqrt(pmax(Omega_diag, 1e-8))

    # Constants for C++
    inv_n1 <- 1 / n1
    inv_n2 <- 1 / n2
    factor1 <- inv_n1 + inv_n2
    factor2 <- rowSums(X_tilde) * inv_n2
    base_indicator <- as.integer(c(rep(1, n1), rep(0, n2)))

    # 2. Get Raw Permutation Matrix (p x n_perms)
    # This matrix contains (mean1^b - mean2^b) for each feature and permutation
    perm_raw_mat <- get_signed_perm_stats_cpp(
        as.matrix(X_tilde), base_indicator,
        as.numeric(factor1), as.numeric(factor2),
        as.integer(n_perms)
    )

    # 3. Compute Feature-wise Null Expectation (E_perm[d_j])
    # null_mean_raw_j: Expected raw difference under null
    null_mean_raw <- rowMeans(perm_raw_mat)

    # 4. Compute Observed Raw Stats
    m1 <- rowMeans(X_tilde[, cluster_labels == 1, drop = FALSE])
    m2 <- rowMeans(X_tilde[, cluster_labels == 2, drop = FALSE])
    obs_raw_signed <- m1 - m2

    # 5. Compute SAM Delta Statistic
    # d_j = raw_signed * scale_factor
    # Delta_j = |d_j - E[d_j]| = |raw_signed - null_mean_raw| * scale_factor

    Delta_obs <- abs(obs_raw_signed - null_mean_raw) * scale_factor_vec

    # 6. Compute Permutation Delta Matrix
    # We need Delta_j^b for all b.
    # Delta_j^b = |raw_signed^b - null_mean_raw| * scale_factor
    # This centers each permutation by the GLOBAL null mean (across all perms).

    perm_centered_raw <- sweep(perm_raw_mat, 1, null_mean_raw, "-")
    Delta_perm_mat <- abs(perm_centered_raw) * scale_factor_vec

    # 7. Grid Search for Delta Threshold
    # Candidates: Sorted Delta_obs
    sorted_deltas <- sort(Delta_obs, decreasing = TRUE)

    # Call C++ to count exceedances efficiently
    # Counts[k] = sum_{b} count(Delta^b >= sorted_deltas[k])
    total_counts <- count_matrix_exceedances_cpp(Delta_perm_mat, sorted_deltas)

    # 8. FDR Calculation
    selected_delta <- 0
    est_fdr <- 0
    best_k <- 0

    # Search for smallest delta (largest k) satisfying FDR target
    for (k in 1:length(sorted_deltas)) {
        delta_val <- sorted_deltas[k]
        if (delta_val <= 0) break

        # R: Number of features with Delta_obs >= delta_val
        R <- k

        # V: Expected number of null features >= delta_val
        V <- total_counts[k] / n_perms

        # FDR
        fdr_val <- V / max(R, 1)

        if (fdr_val <= fdr_target) {
            best_k <- k
            est_fdr <- fdr_val
            selected_delta <- delta_val
        }
    }

    if (best_k == 0) {
        # Fallback
        return(list(delta = Inf, n_selected = 0, est_fdr = NA, s_hat = integer(0)))
    }

    # Identify selected features
    s_hat <- which(Delta_obs >= selected_delta)

    return(list(delta = selected_delta, n_selected = length(s_hat), est_fdr = est_fdr, s_hat = s_hat))
}


#' Deterministic Iterative SDP K-Means with Permutation FDR
#'
#' @param X Data matrix (p x n)
#' @param K Number of clusters
#' @param n_iter Maximum number of iterations
#' @param n_perms Number of permutations for FDR control (default 5000)
#' @param fdr_target Target FDR level (default 0.4)
#' @export
cluster_greedy_ISEE <- function(X, K, n_iter = 200, n_perms = 5000, fdr_target = 0.4, stable_iter = 5, true_labels = NULL) {
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

        # --- Permutation Thresholding (SAM Delta) ---
        cat(sprintf("Estimating Delta Threshold via %d permutations...\n", n_perms))
        # Note: stats_obs argument is ignored inside now, but we keep signature or pass Delta_obs if needed?
        # Actually the function re-calculates everything from X_tilde.
        # So first arg is just a placeholder or we should clean it up.
        # But for minimal disruption, we pass abs_diff (ignored) or pass NULL.

        perm_res <- get_permutation_fdr_threshold(abs_diff, X_tilde, cluster_est_now, Omega_diag_hat, n_perms, fdr_target)

        current_delta <- perm_res$delta
        cat(sprintf("  Selected Delta: %.4f\n", current_delta))
        cat(sprintf("  Features Selected: %d (Est FDR: %.4f)\n", perm_res$n_selected, perm_res$est_fdr))

        s_hat <- perm_res$s_hat

        if (length(s_hat) == 0) {
            cat("  WARNING: No features selected. Using Top K features fallback.\n")
            # Fallback based on raw abs_diff?
            s_hat <- order(abs_diff, decreasing = TRUE)[1:K]
        }

        # --- Clustering Block ---
        # Re-cluster using SDP with selected features
        cat("Running Clustering Block...\n")

        # Estimate Sigma_hat on s_hat
        Sigma_hat_small <- get_cov_small(X, cluster_est_now, s_hat)
        Sigma_full <- diag(1, p)
        if (length(s_hat) > 0) {
            Sigma_full[s_hat, s_hat] <- Sigma_hat_small
        }

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
    # final ISEE refinement
    res_isee <- ISEE_residual_lasso(X, cluster_est_now, K)
    X_tilde_final <- res_isee$X_tilde

    total_time <- difftime(Sys.time(), start_time, units = "secs")

    return(list(
        cluster = cluster_est_now,
        s_hat = s_hat,
        iternum = iternum,
        abs_diff = abs_diff,
        final_delta = current_delta,
        X_tilde_final = X_tilde_final
    ))
}
