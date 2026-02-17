# ISEE Bicluster Algorithm for Unknown Covariance

# source("get_intercept_residual_lasso.R") # Sourced by driver script

#' ISEE Bicluster Algorithm
#'
#' Estimates means and noise using blockwise Lasso regressions.
#'
#' @param x Data matrix (p x n)
#' @param cluster_est_now Cluster assignments (vector of length n)
#' @return List containing:
#'   \item{X_tilde}{Estimated X_tilde matrix (p x n)}
#'   \item{Omega_diag_hat}{Estimated diagonal of precision matrix (p x 1)}
#' @export
ISEE_bicluster <- function(x, cluster_est_now) {
  # Ensure foreach is available
  if (!requireNamespace("foreach", quietly = TRUE)) {
    stop("ISEE_bicluster requires the 'foreach' package.")
  }

  p <- nrow(x)
  n <- ncol(x)
  K <- length(unique(cluster_est_now))

  # Blockwise settings: Block size 2
  n_regression <- floor(p / 2)

  # Transpose X once for efficient row access (columns in x_t)
  x_t <- t(x)

  cat(sprintf("Running ISEE Bicluster on %d blocks (Parallel)...\n", n_regression))

  # Parallel Loop over blocks
  results_list <- foreach::foreach(
    i = 1:n_regression,
    .packages = c("Matrix", "glmnet"),
    .export = c("get_intercept_residual_lasso")
  ) %dopar% {
    # Define block indices (2 rows per block)
    i1 <- 2 * i - 1
    i2 <- 2 * i
    rows_idx <- c(i1, i2)

    # Predictors: All variables except the current block
    predictor_all <- x_t[, -rows_idx, drop = FALSE]

    # Initialize local results
    E_Al <- matrix(0, nrow = 2, ncol = n)
    alpha_Al <- matrix(0, nrow = 2, ncol = K)

    # Loop over clusters
    for (c in 1:K) {
      cluster_mask <- (cluster_est_now == c)
      if (sum(cluster_mask) < 2) next

      # Subset data for this cluster
      predictor_cluster <- predictor_all[cluster_mask, , drop = FALSE]

      # For each variable in the block
      for (j in 1:2) {
        row_id <- rows_idx[j]
        response_cluster <- x_t[cluster_mask, row_id]

        # Run Lasso
        res <- get_intercept_residual_lasso(response_cluster, predictor_cluster)

        # Store results
        E_Al[j, cluster_mask] <- res$residual
        alpha_Al[j, c] <- res$intercept
      }
    }

    # Estimate Omega (Precision) for this block
    scatter_mat <- tcrossprod(E_Al)

    # Handle singularity
    Omega_hat_Al <- tryCatch(
      {
        solve(scatter_mat) * n
      },
      error = function(e) {
        diag(1 / diag(scatter_mat)) * n
      }
    )

    # Logic & Simplification: Direct Computation of X_tilde_local
    # X_tilde_Al = Omega_hat_Al %*% (alpha_Al (broadcasted) + E_Al)

    # Construct (alpha_Al + E_Al)
    # alpha_Al is 2 x K. E_Al is 2 x n.
    # We need to broadcast alpha_Al to 2 x n based on cluster_est_now.
    # Efficiently: We can iterate clusters again or just use indexing.
    # Indexing: alpha_expanded = alpha_Al[, cluster_est_now]

    # However, to be memory efficient, we can sum directly?
    # Summing alpha and E:
    temp_sum <- E_Al
    # Add alpha to each column based on cluster
    # Vectorized: temp_sum + alpha_Al[, cluster_est_now]
    # This works efficiently in R.
    temp_sum <- temp_sum + alpha_Al[, cluster_est_now]

    # Compute X_tilde local block
    X_tilde_local <- Omega_hat_Al %*% temp_sum # 2 x n

    diag_local <- diag(Omega_hat_Al) # 2 x 1

    # Return list for this block
    list(
      rows_idx = rows_idx,
      X_tilde_local = X_tilde_local,
      diag_local = diag_local
    )
  }

  # Reconstruct global X_tilde
  X_tilde <- matrix(0, nrow = p, ncol = n)
  Omega_diag_hat <- numeric(p)

  for (res in results_list) {
    rows_idx <- res$rows_idx
    X_tilde[rows_idx, ] <- res$X_tilde_local
    Omega_diag_hat[rows_idx] <- res$diag_local
  }

  # Handle odd p (final row)
  if (p %% 2 != 0) {
    last_idx <- p
    rows_idx <- last_idx

    predictor_all <- x_t[, -rows_idx, drop = FALSE]

    E_Al <- numeric(n)
    alpha_Al <- numeric(K)

    for (c in 1:K) {
      cluster_mask <- (cluster_est_now == c)
      if (sum(cluster_mask) < 2) next

      predictor_cluster <- predictor_all[cluster_mask, , drop = FALSE]
      response_cluster <- x_t[cluster_mask, rows_idx]

      res <- get_intercept_residual_lasso(response_cluster, predictor_cluster)

      E_Al[cluster_mask] <- res$residual
      alpha_Al[c] <- res$intercept
    }

    scatter_val <- sum(E_Al^2)
    Omega_hat_Al <- if (scatter_val > 1e-8) (n / scatter_val) else 0

    # Scalar logic for X_tilde
    # alpha_Al is 1 x K, E_Al is 1 x n
    temp_sum <- E_Al + alpha_Al[cluster_est_now]
    X_tilde_local <- Omega_hat_Al * temp_sum

    X_tilde[rows_idx, ] <- X_tilde_local
    Omega_diag_hat[rows_idx] <- Omega_hat_Al
  }

  return(list(
    X_tilde = X_tilde,
    Omega_diag_hat = Omega_diag_hat
  ))
}
