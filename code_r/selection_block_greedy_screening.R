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
  
  # 2. Vectorized Permutation Test
  cat(sprintf("Running permutation test (%d permutations)...\n", n_perms))
  
  # Total sum per feature is constant across permutations
  # Sum_total = n1*mu1 + n2*mu2
  # Diff = mu1 - mu2 = (Sum1/n1) - ((Sum_total - Sum1)/n2)
  #      = Sum1 * (1/n1 + 1/n2) - Sum_total/n2
  
  sum_total <- rowSums(X_tilde)
  inv_n1 <- 1/n_g1
  inv_n2 <- 1/n_g2
  factor1 <- inv_n1 + inv_n2
  factor2 <- sum_total * inv_n2
  
  # Generate permutation matrix M (n x n_perms)
  # Each column is a permutation of the indicator vector for group 1
  # The indicator vector has n_g1 ones and n_g2 zeros
  base_indicator <- c(rep(1, n_g1), rep(0, n_g2))
  
  # Create large matrix of indicators - might be memory intensive if n_perms is huge
  # Use replicate to create n x n_perms matrix
  # For p=5000, n=200, n_perms=10000 -> 200 * 10000 = 2e6 elements (small)
  M <- replicate(n_perms, sample(base_indicator))
  
  # Compute Sum1 for all permutations: X_tilde %*% M
  # (p x n) %*% (n x n_perms) -> (p x n_perms)
  # 5000 x 200 * 200 x 10000 -> 5000 x 10000 = 5e7 elements (~400MB)
  # This is manageable.
  sum1_perms <- X_tilde %*% M
  
  # Compute stats
  # stat = abs( factor1 * sum1 - factor2 )
  # We need to broadcast factor2 (length p) across columns
  perm_stats <- abs(sum1_perms * factor1 - factor2)
  
  # 3. Calculate P-values
  # For each feature, count how many perm stats are >= obs stat
  # (Adding 1 for pseudo-count to strictly avoid 0 p-value)
  # Compare perm_stats (p x n_perms) with obs_stat (p vector)
  counts <- rowSums(perm_stats >= obs_stat)
  p_values <- (counts + 1) / (n_perms + 1)
  
  # 4. BH Adjustment
  adj_p_values <- p.adjust(p_values, method = "BH")
  
  # 5. Application of FDR threshold or Raw P-value
  if (!is.null(fdr_level)) {
    # FDR Control (BH)
    selected <- adj_p_values <= fdr_level
    n_selected <- sum(selected)
    cat(sprintf("%d entries survived (FDR: %.2f) | Min adj-p: %.4e | P-val method: Permutation (%d)\n", 
                n_selected, fdr_level, min(adj_p_values), n_perms))
  } else {
    # Raw P-value Threshold (0.01) matches MATLAB
    selected <- p_values <= 0.01
    n_selected <- sum(selected)
    cat(sprintf("%d entries survived (P-val < 0.01) | Min raw-p: %.4e | P-val method: Permutation (%d)\n", 
                n_selected, min(p_values), n_perms))
  }
  
  # Fallback
  if (n_selected == 0) {
    warning("No features selected by BH procedure. Selecting top 1 feature by p-value.")
    min_p_idx <- which.min(p_values)
    selected[min_p_idx] <- TRUE
  }
  
  return(selected)
}
