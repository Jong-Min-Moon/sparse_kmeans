# Clustering Block for Unknown Covariance

source("sdp_kmeans.R")
source("get_cov_small.R")

#' Clustering Block for Unknown Covariance
#' 
#' Corresponds to cluster_SDP_noniso in MATLAB.
#' Uses estimated covariance on selected features to construct affinity matrix.
#' 
#' @param x Original data matrix (p x n)
#' @param K Number of clusters
#' @param mean_now Innovated mean matrix (p x n)
#' @param noise_now Innovated noise matrix (p x n)
#' @param cluster_est_prev Previous cluster assignments (n vector)
#' @param s_hat Logical vector of selected features (p vector)
#' @return New cluster assignments (n vector)
#' @export
run_clustering_block_unknowncov <- function(x, K, mean_now, noise_now, cluster_est_prev, s_hat) {
  
  n <- ncol(x)
  
  # Check if any features selected
  if (sum(s_hat) == 0) {
      warning("No features selected in unknown covariance clustering. Returning previous.")
      return(cluster_est_prev)
  }
  
  # Estimate Sigma_hat on selected features
  # get_cov_small returns s x s matrix
  Sigma_hat_s_hat_now <- get_cov_small(x, cluster_est_prev, s_hat)
  
  # Compute x_tilde (innovated data)
  x_tilde_now <- mean_now + noise_now
  
  # Subset x_tilde to s_hat
  x_tilde_now_s <- x_tilde_now[s_hat, , drop = FALSE]
  
  # Construct Affinity Matrix
  # G = X_tilde_s' * Sigma_hat_s * X_tilde_s
  # Dimension: (n x s) * (s x s) * (s x n) -> n x n
  affinity_matrix <- crossprod(x_tilde_now_s, Sigma_hat_s_hat_now) %*% x_tilde_now_s
  
  # Run SDP K-Means
  # Normalize by n as per MATLAB code?
  # MATLAB: kmeans_sdp_pengwei( affinity_matrix/ n, K);
  # My sdp_kmeans usually takes G. Scaling by 1/n might affect lambda in SDP?
  # In sdp_kmeans.R, do I scale?
  # The constraint is diag(Z) <= 1/n? No, Z 1 = 1.
  # If G is scaled by n, objective is scaled.
  # Let's follow MATLAB scaling to be safe.
  
  cluster_est_new <- sdp_kmeans(affinity_matrix / n, K)
  
  return(cluster_est_new)
}
