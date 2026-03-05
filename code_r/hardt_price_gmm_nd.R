#' Estimate the scale of the d-dimensional variance 
#' 
#' Following Lemma 3.2 in Hardt & Price, this computes a parameter sigma^2 
#' up to a constant multiplicative error by partitioning samples and using the median-of-means estimator.
#' The target metric Var(F) is bounded by ~3x the maximum coordinate variance.
#' 
#' @param X an n x d matrix of samples
#' @param delta confidence parameter (e.g., 0.05)
#' @return A robust estimate of the maximum coordinate variance
estimate_d_dim_variance <- function(X, delta = 0.05) {
  X <- as.matrix(X)
  n <- nrow(X)
  d <- ncol(X)
  
  # Number of groups O(log(1/delta))
  # Using a multiplier of 3 for median robustness
  n_groups <- max(1, ceiling(3 * log(1 / delta)))
  
  if (n_groups >= n / 2) {
    # If the sample size is extremely small, just return the standard max sample variance
    vars <- apply(X, 2, var)
    return(max(vars))
  }
  
  # Shuffle indices for randomly chosen blocks
  idx <- sample(n)
  
  group_size <- floor(n / n_groups)
  
  group_vars <- matrix(0, nrow = n_groups, ncol = d)
  
  for (g in 1:n_groups) {
    # Extract block
    start_idx <- (g - 1) * group_size + 1
    # Last group gets remaining elements if there's an uneven split
    end_idx <- ifelse(g == n_groups, n, g * group_size)
    
    block_idx <- idx[start_idx:end_idx]
    block_data <- X[block_idx, , drop = FALSE]
    
    # Compute empirical variance of each coordinate in this block
    # (using sample variance which is an unbiased estimator)
    group_vars[g, ] <- apply(block_data, 2, var)
  }
  
  # Median of means (variances) across blocks for each dimension
  robust_vars <- apply(group_vars, 2, median)
  
  # Return the maximum variance found across all coordinates
  return(max(robust_vars))
}
