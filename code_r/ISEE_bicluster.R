#' ISEE Bicluster (Original - Separate Slopes per Cluster)
#'
#' NOTE: This is the original implementation kept for reference.
#' For best performance, use ISEE_bicluster() which now uses Post-Lasso.
#'
#' This function estimates the inverse covariance matrix and signal assuming
#' SEPARATE slopes for each cluster (theoretically incorrect).
#'
#' @param x Data matrix (p x n)
#' @param cluster_est_now Cluster assignments (length n vector)
#' @return List containing:
#'   \item{X_tilde}{Estimated X_tilde (p x n)}
#'   \item{Omega_diag_hat}{Diagonal of precision matrix (p x 1)}
#' @export
ISEE_bicluster_original <- function(x, cluster_est_now) {
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

    # alpha_Al is 2 x K. E_Al is 2 x n.
    temp_sum <- E_Al + alpha_Al[, cluster_est_now]

    # Compute X_tilde local block
    X_tilde_local <- Omega_hat_Al %*% temp_sum # 2 x n
    diag_local <- diag(Omega_hat_Al) # 2 x 1

    list(
      rows_idx = rows_idx,
      X_tilde_local = X_tilde_local,
      diag_local = diag_local
    )
  }

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
    temp_sum <- E_Al + alpha_Al[cluster_est_now]
    X_tilde[rows_idx, ] <- Omega_hat_Al * temp_sum
    Omega_diag_hat[rows_idx] <- Omega_hat_Al
  }

  return(list(X_tilde = X_tilde, Omega_diag_hat = Omega_diag_hat))
}

#' ISEE Bicluster Stacked (Optimized)
#'
#' Estimates means and noise using Stacked Lasso regressions (shared slopes across clusters).
#'
#' @param x Data matrix (p x n)
#' @param cluster_est_now Cluster assignments (vector of length n)
#' @return List containing:
#'   \item{X_tilde}{Estimated X_tilde matrix (p x n)}
#'   \item{Omega_diag_hat}{Estimated diagonal of precision matrix (p x 1)}
#' @export
ISEE_bicluster_stacked <- function(x, cluster_est_now) {
  if (!requireNamespace("glmnet", quietly = TRUE)) stop("glmnet required")
  if (!requireNamespace("foreach", quietly = TRUE)) stop("foreach required")

  p <- nrow(x)
  n <- ncol(x)
  x_t <- t(x)
  K <- length(unique(cluster_est_now))

  # Create indicator matrix Z (n x K)
  Z <- matrix(0, n, K)
  for (k in 1:K) {
    Z[cluster_est_now == k, k] <- 1
  }

  n_regression <- floor(p / 2)
  cat(sprintf("Running ISEE Bicluster Stacked on %d blocks (Parallel)...\n", n_regression))

  # Pre-calculate foldid for consistent CV across blocks and tasks
  set.seed(42)
  fold_id <- sample(rep(seq(10), length.out = n))
  # Pre-construct design matrix base
  D_full <- cbind(Z, x_t)

  results_list <- foreach::foreach(
    i = 1:n_regression,
    .packages = c("Matrix", "glmnet")
  ) %dopar% {
    i1 <- 2 * i - 1
    i2 <- 2 * i
    rows_idx <- c(i1, i2)

    # Response Y (n x 2)
    Y_Al <- x_t[, rows_idx, drop = FALSE]

    # Predictors: Indicators Z and all other variables X_Ac
    # Indicators are first K columns. x_t starts at K+1.
    D_mat <- D_full[, -(K + rows_idx), drop = FALSE]

    # Penalty Factor: 0 for indicators (unpenalized), 1 for variables
    p_fac <- c(rep(0, K), rep(1, p - 2))

    # Fit mgaussian Lasso (shared B across response variables)
    cv_fit <- tryCatch(
      {
        glmnet::cv.glmnet(
          x = D_mat, y = Y_Al, family = "mgaussian",
          penalty.factor = p_fac, intercept = FALSE,
          parallel = FALSE, foldid = fold_id
        )
      },
      error = function(e) {
        return(NULL)
      }
    )

    if (is.null(cv_fit)) {
      # Fallback to mean imputation if Lasso fails
      alpha_Al <- matrix(0, 2, K)
      for (k in 1:K) {
        mask <- (cluster_est_now == k)
        if (sum(mask) > 0) alpha_Al[, k] <- colMeans(Y_Al[mask, , drop = FALSE])
      }
      E_Al_t <- Y_Al - Z %*% t(alpha_Al)
    } else {
      # Extract coefficients at lambda.1se
      coef_list <- glmnet::coef.glmnet(cv_fit, s = "lambda.1se")
      # Extract intercepts alpha (K x 2)
      alpha_Al_t <- matrix(0, K, 2)
      # shared beta across y1, y2
      beta_Al_t <- matrix(0, (p - 2), 2)

      for (j in 1:2) {
        # coefs[[j]] is (K + p - 2 + 1) x 1. Row 1 is global intercept (0).
        alpha_Al_t[, j] <- as.numeric(coef_list[[j]][2:(K + 1)])
        beta_Al_t[, j] <- as.numeric(coef_list[[j]][(K + 2):(K + p - 1)])
      }

      # Residuals: E_Al (n x 2) = Y - (Z*alpha + X_Ac*beta)
      # Predictor_Ac columns in D_mat start after first K columns
      E_Al_t <- Y_Al - (Z %*% alpha_Al_t + D_mat[, (K + 1):(K + p - 2)] %*% beta_Al_t)
      alpha_Al <- t(alpha_Al_t) # 2 x K
    }

    # Estimate Omega (2 x 2)
    E_Al <- t(E_Al_t) # 2 x n
    scatter_mat <- tcrossprod(E_Al)

    Omega_hat_Al <- tryCatch(
      {
        solve(scatter_mat) * n
      },
      error = function(e) {
        diag(1 / diag(scatter_mat)) * n
      }
    )

    # X_tilde_Al = Omega_hat_Al %*% (alpha_Al[, cluster_est_now] + E_Al)
    X_tilde_local <- Omega_hat_Al %*% (alpha_Al[, cluster_est_now] + E_Al)
    diag_local <- diag(Omega_hat_Al)

    list(rows_idx = rows_idx, X_tilde_local = X_tilde_local, diag_local = diag_local)
  }

  X_tilde <- matrix(0, nrow = p, ncol = n)
  Omega_diag_hat <- numeric(p)

  for (res in results_list) {
    X_tilde[res$rows_idx, ] <- res$X_tilde_local
    Omega_diag_hat[res$rows_idx] <- res$diag_local
  }

  # Final row if p is odd
  if (p %% 2 != 0) {
    last_idx <- p
    Y_Al <- x_t[, last_idx, drop = FALSE]
    D_mat <- D_full[, -(K + last_idx), drop = FALSE]
    p_fac <- c(rep(0, K), rep(1, p - 1))

    cv_fit <- tryCatch(
      {
        glmnet::cv.glmnet(
          x = D_mat, y = Y_Al, family = "gaussian",
          penalty.factor = p_fac, intercept = FALSE,
          parallel = FALSE, foldid = fold_id
        )
      },
      error = function(e) {
        return(NULL)
      }
    )

    if (is.null(cv_fit)) {
      alpha_Al <- numeric(K)
      for (k in 1:K) {
        mask <- (cluster_est_now == k)
        if (sum(mask) > 0) alpha_Al[k] <- mean(Y_Al[mask])
      }
      E_Al <- Y_Al - Z %*% alpha_Al
    } else {
      coefs <- glmnet::coef.glmnet(cv_fit, s = "lambda.1se")
      alpha_Al <- as.numeric(coefs[2:(K + 1)])
      beta_Al <- as.numeric(coefs[(K + 2):(K + p)])
      E_Al <- Y_Al - (Z %*% alpha_Al + D_mat[, (K + 1):(K + p - 1)] %*% beta_Al)
    }

    scatter_val <- sum(E_Al^2)
    Omega_hat_Al <- if (scatter_val > 1e-8) (n / scatter_val) else 0
    X_tilde[last_idx, ] <- Omega_hat_Al * (E_Al + alpha_Al[cluster_est_now])
    Omega_diag_hat[last_idx] <- Omega_hat_Al
  }

  return(list(X_tilde = X_tilde, Omega_diag_hat = Omega_diag_hat))
}

#' ISEE Bicluster Post-Lasso (Two-Stage: Lasso Selection + OLS Refit)
#'
#' Stage 1: Use Lasso to select support. Stage 2: Refit OLS on selected support.
#'
#' @param x Data matrix (p x n)
#' @param cluster_est_now Cluster assignments (vector of length n)
#' @return List containing:
#'   \item{X_tilde}{Estimated X_tilde matrix (p x n)}
#'   \item{Omega_diag_hat}{Estimated diagonal of precision matrix (p x 1)}
#' @export
ISEE_bicluster_postlasso <- function(x, cluster_est_now) {
  if (!requireNamespace("glmnet", quietly = TRUE)) stop("glmnet required")
  if (!requireNamespace("foreach", quietly = TRUE)) stop("foreach required")

  p <- nrow(x)
  n <- ncol(x)
  x_t <- t(x)
  K <- length(unique(cluster_est_now))

  # Create indicator matrix Z (n x K)
  Z <- matrix(0, n, K)
  for (k in 1:K) {
    Z[cluster_est_now == k, k] <- 1
  }

  n_regression <- floor(p / 2)
  cat(sprintf("Running ISEE Bicluster Post-Lasso on %d blocks (Parallel)...\n", n_regression))

  # Pre-calculate foldid for consistent CV
  set.seed(42)
  fold_id <- sample(rep(seq(10), length.out = n))

  # Pre-construct design matrix base
  D_full <- cbind(Z, x_t)

  results_list <- foreach::foreach(
    i = 1:n_regression,
    .packages = c("Matrix", "glmnet")
  ) %dopar% {
    i1 <- 2 * i - 1
    i2 <- 2 * i
    rows_idx <- c(i1, i2)

    # Response Y (n x 2)
    Y_Al <- x_t[, rows_idx, drop = FALSE]

    # Full design matrix (excluding current block)
    D_mat <- D_full[, -(K + rows_idx), drop = FALSE]

    # Penalty Factor: 0 for indicators, 1 for variables
    p_fac <- c(rep(0, K), rep(1, p - 2))

    # === STAGE 1: Lasso for Support Selection ===
    cv_fit <- tryCatch({
      glmnet::cv.glmnet(x = D_mat, y = Y_Al, family = "mgaussian", 
                        penalty.factor = p_fac, intercept = FALSE, 
                        parallel = FALSE, foldid = fold_id)
    }, error = function(e) return(NULL))

    if (is.null(cv_fit)) {
      # Fallback: Use all variables
      support <- 1:(p-2)
    } else {
      # Extract support from first response variable
      coef_list <- glmnet::coef.glmnet(cv_fit, s = "lambda.1se")
      beta_lasso <- as.numeric(coef_list[[1]][(K+2):(K+p-1)])
      support <- which(beta_lasso != 0)
    }

    # === STAGE 2: OLS Refit on Selected Support ===
    if (length(support) == 0) {
      # No variables selected, use cluster means only
      alpha_Al <- matrix(0, 2, K)
      for(k in 1:K) {
        mask <- (cluster_est_now == k)
        if(sum(mask) > 0) alpha_Al[, k] <- colMeans(Y_Al[mask, , drop = FALSE])
      }
      E_Al_t <- Y_Al - Z %*% t(alpha_Al)
    } else {
      # Build reduced design matrix: Z + selected variables
      D_refit <- cbind(Z, D_mat[, K + support, drop = FALSE])
      
      # Fit OLS for each response variable
      alpha_Al_t <- matrix(0, K, 2)
      beta_Al_t <- matrix(0, length(support), 2)
      
      for(j in 1:2) {
        # OLS: (D'D)^{-1} D'y
        fit_ols <- lm.fit(D_refit, Y_Al[, j])
        coefs_ols <- fit_ols$coefficients
        
        alpha_Al_t[, j] <- coefs_ols[1:K]
        if(length(support) > 0) {
          beta_Al_t[, j] <- coefs_ols[(K+1):(K+length(support))]
        }
      }
      
      # Compute residuals
      E_Al_t <- Y_Al - (Z %*% alpha_Al_t + D_mat[, K + support, drop = FALSE] %*% beta_Al_t)
      alpha_Al <- t(alpha_Al_t)
    }

    # Estimate Omega (2 x 2)
    E_Al <- t(E_Al_t)
    scatter_mat <- tcrossprod(E_Al)

    Omega_hat_Al <- tryCatch({
      solve(scatter_mat) * n
    }, error = function(e) {
      diag(1 / diag(scatter_mat)) * n
    })

    X_tilde_local <- Omega_hat_Al %*% (alpha_Al[, cluster_est_now] + E_Al)
    diag_local <- diag(Omega_hat_Al)

    list(rows_idx = rows_idx, X_tilde_local = X_tilde_local, diag_local = diag_local)
  }

  X_tilde <- matrix(0, nrow = p, ncol = n)
  Omega_diag_hat <- numeric(p)

  for (res in results_list) {
    X_tilde[res$rows_idx, ] <- res$X_tilde_local
    Omega_diag_hat[res$rows_idx] <- res$diag_local
  }

  # Handle odd p (final row)
  if (p %% 2 != 0) {
    last_idx <- p
    Y_Al <- x_t[, last_idx, drop = FALSE]
    D_mat <- D_full[, -(K + last_idx), drop = FALSE]
    p_fac <- c(rep(0, K), rep(1, p - 1))
    
    # Stage 1: Lasso
    cv_fit <- tryCatch({
      glmnet::cv.glmnet(x = D_mat, y = Y_Al, family = "gaussian", 
                        penalty.factor = p_fac, intercept = FALSE, 
                        parallel = FALSE, foldid = fold_id)
    }, error = function(e) return(NULL))
    
    if (is.null(cv_fit)) {
      support <- 1:(p-1)
    } else {
      coefs <- glmnet::coef.glmnet(cv_fit, s = "lambda.1se")
      beta_lasso <- as.numeric(coefs[(K+2):(K+p)])
      support <- which(beta_lasso != 0)
    }
    
    # Stage 2: OLS Refit
    if (length(support) == 0) {
      alpha_Al <- numeric(K)
      for(k in 1:K) {
        mask <- (cluster_est_now == k)
        if(sum(mask) > 0) alpha_Al[k] <- mean(Y_Al[mask])
      }
      E_Al <- Y_Al - Z %*% alpha_Al
    } else {
      D_refit <- cbind(Z, D_mat[, K + support, drop = FALSE])
      fit_ols <- lm.fit(D_refit, Y_Al)
      coefs_ols <- fit_ols$coefficients
      
      alpha_Al <- coefs_ols[1:K]
      beta_Al <- if(length(support) > 0) coefs_ols[(K+1):(K+length(support))] else numeric(0)
      E_Al <- Y_Al - (Z %*% alpha_Al + D_mat[, K + support, drop = FALSE] %*% beta_Al)
    }
    
    scatter_val <- sum(E_Al^2)
    Omega_hat_Al <- if (scatter_val > 1e-8) (n / scatter_val) else 0
    X_tilde[last_idx, ] <- Omega_hat_Al * (E_Al + alpha_Al[cluster_est_now])
    Omega_diag_hat[last_idx] <- Omega_hat_Al
  }

  return(list(X_tilde = X_tilde, Omega_diag_hat = Omega_diag_hat))
}
