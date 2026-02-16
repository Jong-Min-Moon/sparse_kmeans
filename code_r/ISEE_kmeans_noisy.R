# ISEE K-Means for Unknown Covariance (Noisy Case)

# source("ISEE_bicluster.R")
# source("select_variable_ISEE_noisy.R")
# source("clustering_block_unknowncov.R")
# source("sdp_kmeans.R")
# library(mclust) is already handled in the functions using it or by the driver.
library(mclust)

#' ISEE K-Means Algorithm (Unknown Covariance, Noisy)
#' 
#' Iterative algorithm for clustering with unknown covariance structure.
#' Steps:
#' 1. Initialization (SDP K-Means on raw data)
#' 2. ISEE Estimation (Biclustering) -> Get Innovated Data components
#' 3. Variable Selection (Noisy)
#' 4. Clustering (SDP on Affinities from estimated covariance)
#' 
#' @param x Data matrix (p x n)
#' @param K Number of clusters
#' @param n_iter Max iterations
#' @param stable_iter Number of stable iterations to stop
#' @return List with cluster assignments and metrics
#' @export
ISEE_kmeans_noisy <- function(x, K, n_iter = 10, stable_iter = 3) {
  
  if (!is.matrix(x)) stop("x must be a matrix")
  
  p <- nrow(x)
  n <- ncol(x)
  
  cat("Running ISEE K-Means Noisy...\n")
  
  # 1. Initialization
  # Standard SDP K-means on X'X (Euclidean distance kernel equivalent)
  G_init <- crossprod(x)
  cluster_est_now <- sdp_kmeans(G_init, K)
  
  # Metrics tracking
  rand_vec <- numeric(n_iter)
  consecutive_stable <- 0
  is_stop <- FALSE
  iternum <- 0
  
  while (!is_stop && iternum < n_iter) {
    iternum <- iternum + 1
    cat(sprintf("\n--- Iteration %d ---\n", iternum))
    iter_start_time <- Sys.time()
    
    # Check collapse
    if (length(unique(cluster_est_now)) < K) {
       warning("Clusters collapsed.")
       break
    }
    
    # 2. ISEE Estimation
    # Returns mean_vec, noise_mat, Omega_diag_hat, mean_mat
    res_isee <- ISEE_bicluster(x, cluster_est_now)
    # Extract needed components
    mean_mat <- res_isee$mean_mat
    noise_mat <- res_isee$noise_mat
    Omega_diag_hat <- res_isee$Omega_diag_hat
    
    # 3. Variable Selection
    # Returns logical vector s_hat
    s_hat <- select_variable_ISEE_noisy(mean_mat, noise_mat, Omega_diag_hat, cluster_est_now)
    
    # 4. Clustering Block
    # Uses Original X, but computes affinity using innovated data info + estimated cov
    cluster_est_new <- run_clustering_block_unknowncov(x, K, mean_mat, noise_mat, cluster_est_now, s_hat)
    
    # Check convergence
    ri <- adjustedRandIndex(cluster_est_new, cluster_est_now)
    rand_vec[iternum] <- ri
    
    cluster_est_now <- cluster_est_new
    
    cat(sprintf("Adjusted Rand Index change: %.4f\n", ri))
    
    if (ri == 1) {
       consecutive_stable <- consecutive_stable + 1
    } else {
       consecutive_stable <- 0
    }
    
    if (consecutive_stable >= stable_iter) {
       cat("Converged (Stable clustering).\n")
       is_stop <- TRUE
    }
  }
  
  return(list(
     cluster = cluster_est_now,
     iter = iternum,
     rand_vec = rand_vec[1:iternum]
  ))
}
