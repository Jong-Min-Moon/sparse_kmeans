# Selection Block for Known Covariance (Greedy Screening)

#' Selection Block for Known Covariance (Greedy Screening)
#'
#' @param X_tilde Data matrix (p x n)
#' @param cluster_est Current cluster assignments (vector of length n)
#' @param fdr_level False Discovery Rate level (default 0.4)
#' @param n_perms Number of permutations for the test (default 10000)
#' @param normalize Logical. Whether to normalize the data (default TRUE)
#' @param ... Additional arguments (ignored, but allowed for compatibility)
#' @return Logical vector of selected features (length p)
reward_thompson <- function(X_tilde, cluster_est, fdr_level = NULL, n_perms = 10000, p_val_threshold = 0.1, ...) {
  p <- nrow(X_tilde)
  n <- ncol(X_tilde)

  n_g1 <- sum(cluster_est == 1)
  n_g2 <- n - n_g1
  min_n <- min(n_g1, n_g2)
  MMD_sensitivity <- 1 / min_n
  r <- 5
  cat("MMD sensitivity: ", MMD_sensitivity, "\n")
  # Check for degenerate clusters
  if (min_n == 0) {
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

  cat("featurewise minmax scaling...\n")
  X_tilde <- t(X_tilde)
  mins <- matrixStats::colMins(X_tilde)
  maxs <- matrixStats::colMaxs(X_tilde)
  ranges <- maxs - mins

  # Avoid division by zero
  ranges[ranges == 0] <- 1

  X_tilde <- scale(X_tilde, center = mins, scale = ranges)
  X_tilde <- t(X_tilde)

  cat("Calculating observed statistics...\n")
  obs_stat <- calc_diff_means(X_tilde, cluster_est)

  # 2. Optimized C++ Permutation Test
  cat(sprintf("Running permutation test (%d permutations) using C++ backend...\n", n_perms))

  # Ensure the C++ library is loaded
  lib_name <- "selection_utils"
  ext <- if (.Platform$OS.type == "windows") ".dll" else ".so"

  possibilities <- c(
    paste0("code_r/", lib_name, ext),
    paste0("../../code_r/", lib_name, ext),
    paste0("../code_r/", lib_name, ext)
  )

  lib_path <- NULL
  for (path_try in possibilities) {
    if (length(path_try) > 0 && !is.na(path_try) && nzchar(path_try) && file.exists(path_try)) {
      lib_path <- path_try
      break
    }
  }

  if (!is.null(lib_path) && length(lib_path) == 1 && nzchar(lib_path) && file.exists(lib_path)) {
    cat(sprintf("Found C++ backend at: %s\n", lib_path))
    if (!(lib_name %in% names(getLoadedDLLs()))) {
      cat(sprintf("Loading DLL: %s\n", lib_path))
      dyn.load(lib_path)
    }
  } else {
    warning(sprintf(
      "C++ backend for screening ('%s') not found in [%s]. Falling back to R implementation.",
      lib_name, paste(possibilities, collapse = ", ")
    ))
  }

  sum_total <- rowSums(X_tilde)
  inv_n1 <- 1 / n_g1
  inv_n2 <- 1 / n_g2
  factor1 <- inv_n1 + inv_n2
  factor2 <- sum_total * inv_n2

  base_indicator <- as.integer(c(rep(1, n_g1), rep(0, n_g2)))

  # Call C++: returns list with p_value and percentile_value
  use_cpp <- !is.null(lib_path) && length(lib_path) == 1 && nzchar(lib_path) && file.exists(lib_path)

  if (use_cpp) {
    res <- .Call(
      "fast_perm_test_wrapper",
      as.matrix(X_tilde),
      as.numeric(obs_stat),
      base_indicator,
      as.numeric(factor1),
      as.numeric(factor2),
      as.integer(n_perms),
      as.numeric(p_val_threshold)
    )
    p_values <- res$p_value
    percentile_val <- res$percentile_value
  } else {
    # Re-implementing original R logic as fallback
    M <- replicate(n_perms, sample(base_indicator))
    sum1_perms <- X_tilde %*% M
    perm_stats <- abs(sum1_perms * factor1 - factor2)
    counts <- rowSums(perm_stats >= obs_stat)
    p_values <- (counts + 1) / (n_perms + 1)

    q <- 1.0 - p_val_threshold
    k <- max(1, min(ceiling(q * n_perms), n_perms))
    percentile_val <- apply(perm_stats, 1, function(x) sort(x, partial = k)[k])
  }


  # Raw P-value Threshold (matching user preference)
  selected <- obs_stat >= (percentile_val + 2 * r * MMD_sensitivity)
  n_selected <- sum(selected)
  cat(sprintf(
    "%d entries survived (P-val < %.4f) | Min raw-p: %.4e | Min percentile: %.5e| P-val method: Permutation (%d)\n",
    n_selected, p_val_threshold, min(p_values), min(percentile_val), n_perms
  ))


  # Fallback
  if (n_selected == 0) {
    warning("No features selected by permutation threshold. Selecting top 1 feature by p-value.")
    min_p_idx <- which.min(p_values)
    selected[min_p_idx] <- TRUE
  }

  return(selected)
}
