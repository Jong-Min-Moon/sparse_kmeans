# Helper to Estimate Covariance on Small Subset

#' Estimate Covariance Matrix for Selected Variables
#' 
#' Computes the sample covariance matrix for the selected variables, assuming
#' equal covariance across clusters (pooled covariance) or common structure.
#' 
#' @param x Data matrix (p x n)
#' @param cluster_est Cluster labels (n vector)
#' @param s_hat Logical vector of selected variables (length p)
#' @return Sigma_hat (s x s matrix)
#' @export
get_cov_small <- function(x, cluster_est, s_hat) {
  
  # Subset data to selected variables
  # s_hat can be logical or indices
  if (is.logical(s_hat)) {
    s_indices <- which(s_hat)
  } else {
    s_indices <- s_hat
  }
  
  if (length(s_indices) == 0) {
      stop("No variables selected for covariance estimation.")
  }
  
  # Dimensions
  # p_small <- length(s_indices)
  n <- ncol(x)
  
  # Extract relevant rows
  x_sub <- x[s_indices, , drop = FALSE]
  
  # Compute pooled covariance
  # Center each group separately
  unique_clusters <- unique(cluster_est)
  
  # Initialize centered matrix
  x_centered <- x_sub
  
  for (k in unique_clusters) {
      idx <- which(cluster_est == k)
      if (length(idx) > 0) {
          # Calculate mean for this cluster
          mu_k <- rowMeans(x_sub[, idx, drop = FALSE])
          # Subtract mean
          x_centered[, idx] <- x_sub[, idx, drop = FALSE] - mu_k
      }
  }
  
  # Calculate covariance of centered data
  # cov() function divides by (n-1)
  # MATLAB code: cov(data_filtered') 
  # MATLAB cov normalizes by N-1 by default.
  # So we just take cov of x_centered (transposed)
  
  Sigma_hat <- cov(t(x_centered))
  
  return(Sigma_hat)
}
