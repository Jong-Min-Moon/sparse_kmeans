# Select Variables using ISEE (Noisy Case)

#' Select Variables using Method for Noisy Data
#' 
#' Selects variables based on the magnitude of the estimated signal relative to noise.
#' 
#' @param mean_now p x n matrix of cluster center part of innovated data (pre-multiplied by precision)
#' @param noise_now p x n matrix of noise part of innovated data
#' @param Omega_diag_hat p vector of diagonal entries of precision matrix
#' @param cluster_est_prev n vector of cluster labels
#' @return Logical vector of length p indicating selected variables
#' @export
select_variable_ISEE_noisy <- function(mean_now, noise_now, Omega_diag_hat, cluster_est_prev) {
  
  # Compute innovated data
  x_tilde_now <- mean_now + noise_now
  
  p <- nrow(mean_now)
  n <- ncol(mean_now)
  
  # Threshold
  thres <- sqrt(2 * log(p))
  
  # Signal estimation difference between groups
  # Assuming cluster labels are 1 and 2
  # Calculate mean of x_tilde for each group
  
  # Vectorized calculation for efficiency
  # x_tilde_g1 <- x_tilde_now[, cluster_est_prev == 1, drop = FALSE]
  # x_tilde_g2 <- x_tilde_now[, cluster_est_prev == 2, drop = FALSE]
  
  # Compute row means
  # signal_est_now <- rowMeans(x_tilde_g1) - rowMeans(x_tilde_g2)
  
  # Optimized with matrix multiplication if n is large, but rowMeans is fast
  # Handle case where cluster might be empty? (Should be handled outside)
  
  idx1 <- which(cluster_est_prev == 1)
  idx2 <- which(cluster_est_prev == 2)
  n_g1 <- length(idx1)
  n_g2 <- length(idx2)
  
  if (n_g1 == 0 || n_g2 == 0) {
      warning("One cluster is empty in selection step.")
      return(rep(FALSE, p))
  }
  
  mu1 <- rowMeans(x_tilde_now[, idx1, drop = FALSE])
  mu2 <- rowMeans(x_tilde_now[, idx2, drop = FALSE])
  signal_est_now <- mu1 - mu2
  
  # Standardize difference
  # abs_diff = abs(signal) / sqrt(Omega_diag) * sqrt( n1*n2 / n )
  # This corresponds to t-statistic like scaling
  
  scaling_factor <- sqrt((n_g1 * n_g2) / n)
  abs_diff <- abs(signal_est_now) / sqrt(Omega_diag_hat) * scaling_factor
  
  # Selection
  s_hat <- abs_diff > thres
  
  # Diagnostics
  num_selected <- sum(s_hat)
  cat(sprintf("%d out of %d variables selected.\n", num_selected, p))
  
  return(s_hat)
}
