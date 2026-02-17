# Iterative SDP K-Means with Known Covariance (Refactored)

# source("sdp_kmeans.R")
# source("utils.R")
# source("selection_block_greedy_screening.R")
# source("clustering_block_knowncov.R")

#' Iterative SDP K-Means with Known Covariance
#'
#' @param X_tilde Data matrix (p x n) (Transformed data)
#' @param K Number of clusters
#' @param n_iter Maximum number of iterations
#' @param stable_iter Number of consecutive iterations with ARI=1 to stop (default 10)
#' @param fdr_level FDR level for selection block (default 0.4)
#' @return List containing cluster assignments, iteration history, and timing
block_coordinate_optim_greedy <- function(X_tilde, K, n_iter = 10, stable_iter = 10, fdr_level = 0.4) {
  if (!is.matrix(X_tilde)) stop("X_tilde must be a matrix")
  p <- nrow(X_tilde)
  n <- ncol(X_tilde)

  start_time <- Sys.time()

  # Initial Clustering (Initialization Block)
  cat("Running initial clustering...\n")
  t_init_start <- Sys.time()
  # Use all features and identity covariance for initialization
  G_init <- crossprod(X_tilde)
  sdp_init <- sdp_kmeans(G_init, K)
  cluster_est_now <- sdp_init$cluster
  t_init_end <- Sys.time()
  cat(sprintf("Initial Clustering took: %.4f seconds\n", as.numeric(difftime(t_init_end, t_init_start, units = "secs"))))

  # Iteration
  is_stop <- FALSE
  iternum <- 0
  rand_vec <- rep(NA, n_iter)

  # Counter for consecutive stable iterations
  consecutive_stable_count <- 0

  while (!is_stop && iternum < n_iter) {
    iternum <- iternum + 1
    cat(sprintf("\n--- Iteration %d ---\n", iternum))
    iter_start_time <- Sys.time()

    # Check convergence of cluster sizes
    if (length(unique(cluster_est_now)) < K) {
      cat("Clusters collapsed to fewer than K groups.\n")
      # Need to account for early return in timing?
      # Usually just breaking here is safer but return is fine.
      return(list(cluster = cluster_est_now, iter = iternum, rand_vec = rand_vec))
    }

    # --- Selection Block ---
    # Now uses BH procedure with fdr_level
    t_sel_start <- Sys.time()
    selected_features <- selection_block_greedy_screening(X_tilde, cluster_est_now, fdr_level)
    t_sel_end <- Sys.time()
    cat(sprintf("Selection Block took: %.4f seconds\n", as.numeric(difftime(t_sel_end, t_sel_start, units = "secs"))))

    # --- Clustering Block ---
    # Using NULL for covariance implies Identity (efficient path)
    t_clus_start <- Sys.time()
    res_clustering <- run_clustering_block_knowncov(X_tilde, selected_features, K, cluster_est_now, covariance = NULL)
    cluster_est_new <- res_clustering$cluster
    t_clus_end <- Sys.time()
    cat(sprintf("Clustering Block took: %.4f seconds\n", as.numeric(difftime(t_clus_end, t_clus_start, units = "secs"))))

    # --- Stopping Criteria ---
    # Compare new clustering with old clustering using Adjusted Rand Index
    ri_result <- RandIndex(cluster_est_new, cluster_est_now)
    rand_score <- ri_result$AR
    rand_vec[iternum] <- rand_score

    iter_end_time <- Sys.time()
    iter_duration <- as.numeric(difftime(iter_end_time, iter_start_time, units = "secs"))
    cat(sprintf("Iteration %d Duration: %.2f seconds\n", iternum, iter_duration))
    cat(sprintf("Adjusted Rand Index change: %.4f\n", rand_score))

    # Stopping logic: Stop if Rand Score is perfectly 1 for 'stable_iter' consecutive times
    if (rand_score == 1) {
      consecutive_stable_count <- consecutive_stable_count + 1
      cat(sprintf("Stable iterations count: %d/%d\n", consecutive_stable_count, stable_iter))
    } else {
      consecutive_stable_count <- 0
    }

    if (consecutive_stable_count >= stable_iter) {
      is_stop <- TRUE
      cat(sprintf("Stopping early: Clustering stable for %d iterations.\n", stable_iter))
    }

    cluster_est_now <- cluster_est_new
  }

  end_time <- Sys.time()
  total_time <- end_time - start_time

  return(list(
    cluster = cluster_est_now,
    iter = iternum,
    rand_vec = rand_vec,
    time = total_time
  ))
}
