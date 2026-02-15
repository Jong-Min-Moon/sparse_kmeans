# Functions for SDP K-Means clustering using CVXR

#' SDP K-Means Clustering
#'
#' Solves the SDP relaxation of K-Means clustering using an affinity matrix.
#'
#' @param G Affinity/Gram matrix (n x n)
#' @param K Number of clusters
#' @param solver Constraints solver (default = "SCS")
#' @return Cluster assignments (vector of length n)
#' @import CVXR
#' @import stats
sdp_kmeans <- function(G, K, solver = "SCS") {
  # 1. Input Validation
  if (!is.matrix(G)) {
    stop("Input G must be a matrix")
  }
  
  n <- nrow(G)
  if (ncol(G) != n) {
    stop("Input G must be a square matrix")
  }
  
  # Ensure K is integer
  K <- as.integer(K)
  
  # 2. Setup CVXR Problem
  # We want to maximize the association, which is equivalent to K-means objective
  # Maximize <G, Z>
  Z <- Variable(n, n, PSD = TRUE)
  objective <- Maximize(sum_entries(G * Z))
  
  constraints <- list(
    Z >= 0,
    sum_entries(Z, axis = 1) == 1,
    matrix_trace(Z) == K
  )
  
  prob <- Problem(objective, constraints)
  
  # 4. Solve
  # Using verbose = FALSE as per snippet
  result <- solve(prob, solver = solver, verbose = FALSE, max_iters = 2500)
  
  # 5. Rounding and Extraction
  # Extract Z
  if (result$status %in% c("optimal", "optimal_inaccurate")) {
    Z_sol <- result$getValue(Z)
    # Ensure symmetry
    Z_sol <- (Z_sol + t(Z_sol)) / 2
    
    # Eigen Decomposition
    # Extract top k eigenvectors
    eigen_decomp <- eigen(Z_sol, symmetric = TRUE)
    # Vectors are in columns. Top K corresponds to largest eigenvalues.
    # eigen() returns sorted eigenvalues descending by default.
    V <- eigen_decomp$vectors[, 1:K]
    
    # K-Means on eigenvectors
    # Using nstart=10 for robustness
    final_clustering <- kmeans(V, centers = K, nstart = 10)
    
    return(final_clustering$cluster)
  } else {
    warning(paste("SDP Solver failed with status:", result$status))
    return(rep(NA, n))
  }
}
