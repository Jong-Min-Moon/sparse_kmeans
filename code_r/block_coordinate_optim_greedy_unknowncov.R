# Iterative SDP K-Means with Known Covariance (Refactored)

# source("sdp_kmeans.R")
# source("utils.R")
# source("selection_block_greedy_screening.R")
# source("clustering_block_knowncov.R")

#' Iterative SDP K-Means with Known Covariance
#'
#' @param X raw Data matrix (p x n)
#' @param K Number of clusters
#' @param n_iter Maximum number of iterations
#' @param stable_iter Number of consecutive iterations with ARI=1 to stop (default 10)
#' @param fdr_level FDR level for selection block (default 0.1)
#' @return List containing cluster assignments, iteration history, and timing
block_coordinate_optim_greedy_unknowncov <- function(X, K, n_iter = 10, stable_iter = 10, fdr_level = 0.4) {
  if (!is.matrix(X)) stop("X must be a matrix")
  p <- nrow(X)
  n <- ncol(X)

  start_time <- Sys.time()

  # Initial Clustering (Initialization Block)
  cat("Running initial clustering...\n")
  # Use all features and identity covariance for initialization
  G_init <- crossprod(X)
  res_init <- sdp_kmeans(G_init, K)
  cluster_est_now <- res_init$cluster

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

    # --- ISEE Estimation ---
    # Estimate denoised matrix X_tilde based on previous clusters
    cat("Running ISEE in Clustering Block...\n")
    res_isee <- ISEE_bicluster(X, cluster_est_now)
    X_tilde <- res_isee$X_tilde

    # --- Selection Block ---
    # Now uses BH procedure with fdr_level
    selected_features <- selection_block_greedy_screening(X_tilde, cluster_est_now, fdr_level)

    selected_indices <- which(selected_features)
    cat(sprintf("Iteration %d: Selected %d features: ", iternum, length(selected_indices)))
    cat(paste(selected_indices, collapse = ", "), "\n")


    # 2. Sub-matrix Extraction
    # Subset X_tilde using selected_features
    X_tilde_sub <- X_tilde[selected_features, , drop = FALSE]

    # 3. Covariance Calculation (Crucial Step)
    # Calculate sample covariance of original x using selected_features
    x_sub <- X[selected_features, , drop = FALSE]

    # Demean per cluster to remove cluster effect (Pooled Covariance estimation)
    x_sub_centered <- x_sub
    unique_clusters <- unique(cluster_est_now)
    for (k in unique_clusters) {
      cluster_idx <- (cluster_est_now == k)
      if (sum(cluster_idx) > 0) {
        cluster_data <- x_sub[, cluster_idx, drop = FALSE]
        cluster_mean <- rowMeans(cluster_data)
        # Subtract cluster mean from each column in that cluster
        x_sub_centered[, cluster_idx] <- sweep(cluster_data, 1, cluster_mean, "-")
      }
    }

    # Calculate covariance of centered data
    cov_sub <- cov(t(x_sub_centered))

    # --- Clustering Block ---
    # Using NULL for covariance implies Identity (efficient path)
    res_blocking <- run_clustering_block_knowncov(
      X_tilde = X_tilde_sub,
      selected_features = seq_len(nrow(X_tilde_sub)),
      K = K,
      cluster_est_prev = cluster_est_now,
      covariance = cov_sub,
      max_iter = 4000
    )
    cluster_est_new <- res_blocking$cluster

    # --- Stopping Criteria ---
    # Compare new clustering with old clustering using Adjusted Rand Index
    rand_score <- mclust::adjustedRandIndex(cluster_est_new, cluster_est_now)
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
    rand_vec = rand_vec,
    time = total_time,
    selected_features = selected_features
  ))
}
