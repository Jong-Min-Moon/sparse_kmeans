# Clustering Block for Known Covariance

# source("sdp_kmeans.R") # Sourced by driver script

#' Clustering Block for Known Covariance
#'
#' @param X_tilde Data matrix (p x n)
#' @param selected_features Logical vector of selected features
#' @param K Number of clusters
#' @param cluster_est_prev Previous cluster assignments (for fallback)
#' @param covariance Covariance structure (matrix). If NULL, assumes Identity.
#' @return New cluster assignments (vector of length n)
run_clustering_block_knowncov <- function(X_tilde, selected_features, K, cluster_est_prev, covariance = NULL) {
  n_selected <- sum(selected_features)

  if (n_selected > 0) {
    # Transformation Step (Subsetting)
    X_sub <- X_tilde[selected_features, , drop = FALSE]

    # Compute Gram/Affinity Matrix
    # G = X_sub' * Cov_sub * X_sub

    # Check if covariance is identity or NULL
    is_identity <- is.null(covariance)
    if (!is_identity && is.matrix(covariance)) {
      # Check dimensions
      if (ncol(covariance) == nrow(X_tilde) && nrow(covariance) == nrow(X_tilde)) {
        cat("Using provided covariance matrix (non-Identity).\n")
        cov_sub <- covariance[selected_features, selected_features, drop = FALSE]
        G <- crossprod(X_sub, cov_sub) %*% X_sub
      } else {
        warning("Covariance dimension mismatch. Using Identity.")
        is_identity <- TRUE
      }
    }

    if (is_identity) {
      cat("Using Identity covariance.\n")
      t_gram_start <- Sys.time()
      G <- crossprod(X_sub)
      t_gram_end <- Sys.time()
      cat(sprintf("  Gram matrix computation took: %.4f seconds\n", as.numeric(difftime(t_gram_end, t_gram_start, units = "secs"))))
    }

    # Clustering Step (SDP)
    t_sdp_start <- Sys.time()
    sdp_res <- sdp_kmeans(G, K)
    cluster_est_new <- sdp_res$cluster
    t_sdp_end <- Sys.time()
    cat(sprintf("  SDP K-Means step took: %.4f seconds\n", as.numeric(difftime(t_sdp_end, t_sdp_start, units = "secs"))))
    # We return the full result to access the objective value in bandit algorithm
    return(sdp_res)
  } else {
    warning("No features selected. Keeping previous clustering.")
    # Return previous with NA value
    return(list(
      cluster = cluster_est_prev,
      value = NA
    ))
  }
}
