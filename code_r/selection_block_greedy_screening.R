# Selection Block for Known Covariance (Greedy Screening)

#' Selection Block for Known Covariance (Greedy Screening)
#'
#' @param X_tilde Data matrix (p x n)
#' @param cluster_est Current cluster assignments (vector of length n)
#' @param fdr_level False Discovery Rate level for BH procedure (default 0.4)
#' @param n_perms Number of permutations for the test (default 10000)
#' @return Logical vector of selected features (length p)
selection_block_greedy_screening <- function(X_tilde, cluster_est, fdr_level = 0.4, n_perms = 10000) {
  p <- nrow(X_tilde)
  n <- ncol(X_tilde)

  n_g1 <- sum(cluster_est == 1)
  n_g2 <- n - n_g1

  # Check for degenerate clusters
  if (n_g1 == 0 || n_g2 == 0) {
    warning("One cluster is empty. Returning all TRUE to avoid crash.")
    return(rep(TRUE, p))
  }

  # Helper to calculate absolute difference in means
  calc_diff_means <- function(X, cl) {
    x_g1 <- X[, cl == 1, drop = FALSE]
    mu1 <- rowMeans(x_g1)

    x_g2 <- X[, cl == 2, drop = FALSE]
    mu2 <- rowMeans(x_g2)

    # Use simple absolute difference as test statistic
    stat <- abs(mu1 - mu2)
    return(stat)
  }

  # 1. Observed Statistic
  cat("Calculating observed statistics...\n")
  obs_stat <- calc_diff_means(X_tilde, cluster_est)

  # 2. Optimized C++ Permutation Test
  cat(sprintf("Running permutation test (%d permutations) using C++ backend...\n", n_perms))

  # Ensure the C++ library is loaded
  # Similar to sdp_kmeans.R, we check for the library
  lib_name <- "selection_utils"
  if (.Platform$OS.type == "windows") {
    lib_path <- paste0("code_r/", lib_name, ".dll")
  } else {
    lib_path <- paste0("code_r/", lib_name, ".so")
  }

  if (!file.exists(lib_path)) {
    # Try relative paths
    alt_paths <- c(paste0("../../", lib_path), paste0("../", lib_path))
    for (ap in alt_paths) {
      if (file.exists(ap)) {
        lib_path <- ap
        break
      }
    }
  }

  if (file.exists(lib_path)) {
    if (!(lib_name %in% names(getLoadedDLLs()))) dyn.load(lib_path)
  } else {
    warning("C++ backend for screening not found. Falling back to (slow) R implementation.")
    # Fallback R implementation (simplified for brevity here, or just keep original)
    # [Original R logic would go here if we wanted robust fallback]
  }

  sum_total <- rowSums(X_tilde)
  inv_n1 <- 1 / n_g1
  inv_n2 <- 1 / n_g2
  factor1 <- inv_n1 + inv_n2
  factor2 <- sum_total * inv_n2

  base_indicator <- as.integer(c(rep(1, n_g1), rep(0, n_g2)))

  # Call C++: returns vector of counts (length p)
  if (file.exists(lib_path)) {
    counts <- .Call(
      "fast_perm_test_wrapper", as.matrix(X_tilde), as.numeric(obs_stat),
      base_indicator, as.numeric(factor1), as.numeric(factor2), as.integer(n_perms)
    )
  } else {
    # Re-implementing original R logic as fallback
    M <- replicate(n_perms, sample(base_indicator))
    sum1_perms <- X_tilde %*% M
    perm_stats <- abs(sum1_perms * factor1 - factor2)
    counts <- rowSums(perm_stats >= obs_stat)
  }

  p_values <- (counts + 1) / (n_perms + 1)

  # 4. BH Adjustment
  adj_p_values <- p.adjust(p_values, method = "BH")

  # 5. Application of FDR threshold or Raw P-value
  if (!is.null(fdr_level)) {
    # FDR Control (BH)
    selected <- adj_p_values <= fdr_level
    n_selected <- sum(selected)
    cat(sprintf(
      "%d entries survived (FDR: %.2f) | Min adj-p: %.4e | P-val method: Permutation (%d)\n",
      n_selected, fdr_level, min(adj_p_values), n_perms
    ))
  } else {
    # Raw P-value Threshold (0.01) matches MATLAB
    selected <- p_values <= 0.01
    n_selected <- sum(selected)
    cat(sprintf(
      "%d entries survived (P-val < 0.01) | Min raw-p: %.4e | P-val method: Permutation (%d)\n",
      n_selected, min(p_values), n_perms
    ))
  }

  # Fallback
  if (n_selected == 0) {
    warning("No features selected by BH procedure. Selecting top 1 feature by p-value.")
    min_p_idx <- which.min(p_values)
    selected[min_p_idx] <- TRUE
  }

  return(selected)
}
