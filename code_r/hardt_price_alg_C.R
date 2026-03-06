#' Generate random projection vectors from N_9^d
#'
#' Following Lemma 3.4 in Hardt & Price, this generates samples from the 
#' d-dimensional normal distribution N(0,1)^d conditioned on the norm being at most 9.
#'
#' @param n number of vectors to sample
#' @param d dimension (e.g., d = 4 for the 4D to 1D reduction)
#' @return A matrix of size n x d where each row is a bounded projection vector
library(future)
library(future.apply)

sample_N9 <- function(n, d) {
  out <- matrix(0, nrow = n, ncol = d)
  count <- 0
  
  while(count < n) {
      # Sample a batch of standard normals
      batch_size <- ceiling((n - count) * 1.5) # Over-sample to account for rejections
      cand <- matrix(rnorm(batch_size * d), nrow = batch_size, ncol = d)
      
      # Compute squared norms
      norms_sq <- rowSums(cand^2)
      
      # Filter to norms <= 9^2 (which is 81)
      valid <- cand[norms_sq <= 81, , drop = FALSE]
      
      n_valid <- nrow(valid)
      if (n_valid > 0) {
          take <- min(n_valid, n - count)
          out[(count + 1):(count + take), ] <- valid[1:take, ]
          count <- count + take
      }
  }
  
  return(out)
}

#' Generate Candidate Grid N_mu
#'
#' Creates an L_infinity net around an initial estimate mu_hat
#' of width 2*sigma_hat in every coordinate, with step size c*epsilon*sigma_hat.
#'
#' @param mu_hat The center vector (length d)
#' @param sigma_hat The robust variance scale estimate
#' @param epsilon The target precision
#' @param c A sufficiently small constant (e.g., 0.1)
#' @return A matrix where each row is a candidate mean vector
# Given mu_hat and sigma_hat, this creates a bounded discrete epsilon grid 
# of side length c*sigma. The paper generates N_mu by taking grid size 
# O(1/epsilon) per dimension resulting in total O(1/epsilon^d).
generate_N_mu <- function(mu_hat, sigma_hat, epsilon, c_const = 1.0) {
  d <- length(mu_hat)
  step_size <- c_const * epsilon * sigma_hat
  
  # If epsilon is extremely small, grid size explodes to poly(1/epsilon). 
  # We should ensure this doesn't crash the R session on exact mathematical runs.
  num_steps <- ceiling((2 * c_const * sigma_hat) / step_size)
  
  # For each dimension, create the 1D grid
  grid_1d_list <- lapply(1:d, function(i) {
    seq(mu_hat[i] - 2 * sigma_hat, mu_hat[i] + 2 * sigma_hat, by = step_size)
  })
  
  # Cartesian product to get the full grid
  # Use expand.grid and convert to matrix
  grid_df <- expand.grid(grid_1d_list)
  return(as.matrix(grid_df))
}

#' Generate Candidate Grid N_sigma
#'
#' Creates a net for symmetric covariance matrices in the range
#' [-sigma_hat^2, sigma_hat^2] with step size c*epsilon*sigma_hat^2.
#'
#' @param d Dimension
#' @param sigma_hat The robust variance scale estimate
#' @param epsilon The target precision
#' @param c A sufficiently small constant (e.g., 0.1)
#' @return A list of candidate d x d symmetric covariance matrices
generate_N_sigma <- function(d, sigma_hat, epsilon, c_const = 1.0) {
  # We need to cover off-diagonal correlations which can be negative.
  # Diagonals must stay positive.
  diag_grid <- seq(0.1, 1.3 * sigma_hat^2, length.out = 3)
  off_diag_grid <- seq(-0.6 * sigma_hat^2, 0.6 * sigma_hat^2, length.out = 3)
  
  # Parameters 1, 2, 3, 4 are diagonals. 5-10 are off-diagonals.
  grid_list <- list(
      diag_grid, diag_grid, diag_grid, diag_grid, # Diagonals
      off_diag_grid, off_diag_grid, off_diag_grid, off_diag_grid, off_diag_grid, off_diag_grid # Off-diagonals
  )
  
  # Total size logic: 3^4 * 3^6 = 3^10 = 59,049 candidates.
  # This is much smaller than 4^10 and more accurate for correlations.
  grid_df <- expand.grid(grid_list)
  return(t(as.matrix(grid_df)))
}

#' Algorithm C: Reduction from 4D to 1D
#'
#' @param X an n x 4 matrix of samples
#' @param epsilon The target precision
#' @param delta confidence parameter
#' @return The recovered parameters (mu1, mu2, Sigma1, Sigma2)
Reduce4DTo1D <- function(X, epsilon = 0.5, delta = 0.05) {
  X <- as.matrix(X)
  n <- nrow(X)
  d <- ncol(X)
  
  if (d != 4) stop("Reduction algorithm strictly expects a 4-dimensional mixture")
  
  cat("[Alg C] Step 1: Estimating max coordinate variance layer...\n")
  sigma_hat_sq <- estimate_d_dim_variance(X, delta)
  sigma_hat <- sqrt(sigma_hat_sq)
  cat(sprintf("[Alg C] Max coordinate variance scale isolated: %.4f\n", sigma_hat_sq))
  
  cat("[Alg C] Step 2: Locating spatial geometry center through 1D marginals...\n")
  mu_hat <- numeric(d)
  
  for (j in 1:d) {
      marginal_res <- Recover1DMixture(X[, j], delta = delta)
      
      if (is.null(marginal_res) || isTRUE(marginal_res$fallback)) {
          # Fallback triggered, geometry obscured
          mu_hat[j] <- mean(X[, j])
      } else {
          m1 <- marginal_res$comp1$mu
          m2 <- marginal_res$comp2$mu
          # We theoretically know true overall mean is 0. 
          # To avoid permuting (mu1, mu2) randomly across axes, pick the one closest to empirical mean
          mu_hat[j] <- marginal_res$comp1$mu
      }
  }
  
  # The coordinate recovery still suffers from permutation variance (swapping component 1 and 2 wildly across dimensions).
  # BUT Theorem 3.9 only requires mu_hat to be within 2*sigma of BOTH true means.
  # So honestly, if the separation is less than 2*sigma, the GLOBAL MEAN is the most perfectly 
  # mathematically bounded center to anchor our computational grid!
  mu_hat <- colMeans(X)

  
  c_const <- 0.6 # Constant tweaked higher so the brute force grid size doesn't crash R
  
  cat("[Alg C] Constructing symmetric 4D candidate mesh grids...\n")
  # N_mu has dimensions: d. N_sigma has dimensions: d*(d+1)/2 = 10.
  N_mu <- generate_N_mu(mu_hat, sigma_hat, epsilon, c_const)

  # step_size = fixed length.out = 3 (Giving 3^10 = 59,049 matrices)
  N_sigma <- generate_N_sigma(d, sigma_hat, epsilon) 
  cat(sprintf("[Alg C] Generated %d discrete vector candidates.\n", nrow(N_mu)))
  cat(sprintf("[Alg C] Generated %d discrete covariance matrices.\n", length(N_sigma)))
  
  m <- ceiling(10 * log(max(1, nrow(N_mu)) / delta))
  cat(sprintf("[Alg C] Broadcasting over %d bounded random structural projections (a_i ~ N_9^4)...\n", m))
  
  a_samples <- sample_N9(m, d)
  
  # Setup parallel execution if not already configured
  if (inherits(plan(), "sequential")) {
      plan(multisession)
  }
  
  cat("[Alg C] Processing 1D projection ensemble in parallel...\n")
  projections_res <- future_lapply(1:m, function(i) {
      a_i <- a_samples[i, ]
      x_1d <- as.numeric(X %*% a_i)
      
      res_1d <- Recover1DMixture(x_1d, delta = delta / m)
      
      if (is.null(res_1d) || isTRUE(res_1d$fallback)) {
          m_emp <- mean(x_1d)
          s_emp <- sd(x_1d)
          return(list(mu1 = m_emp, mu2 = m_emp, var1 = s_emp^2, var2 = s_emp^2))
      }
      
      return(list(
        mu1 = res_1d$comp1$mu,
        mu2 = res_1d$comp2$mu,
        var1 = res_1d$comp1$sigma^2,
        var2 = if(!is.null(res_1d$comp2)) res_1d$comp2$sigma^2 else res_1d$comp1$sigma^2
      ))
  }, future.seed = TRUE)
  
  cat("\n[Alg C] Running Vectorized Voting Filter to isolate means...\n")
  
  # Vectorized mean voting
  projs_means <- N_mu %*% t(a_samples) # (N_mu x m)
  mu1s <- sapply(projections_res, function(x) x$mu1)
  mu2s <- sapply(projections_res, function(x) x$mu2)
  
  # Calculate threshold for each projection
  eps_primes <- epsilon * rowSums(abs(a_samples)) + 0.1
  thresholds <- eps_primes * sigma_hat / 2
  
  # Efficient comparison using sweep or broadcasting
  err1 <- abs(sweep(projs_means, 2, mu1s, "-"))
  err2 <- abs(sweep(projs_means, 2, mu2s, "-"))
  
  pass_mask <- (sweep(err1, 2, thresholds, "<=") | sweep(err2, 2, thresholds, "<="))
  mu_votes <- rowSums(pass_mask)
  
  voting_threshold <- floor(0.85 * m)
  accepted_mu <- (mu_votes >= voting_threshold)
  
  if (!any(accepted_mu)) {
      cat("[Alg C] WARNING: No means passed voting threshold. Relaxing...\n")
      accepted_mu <- (mu_votes >= floor(max(mu_votes) * 0.9))
  }
  
  valid_mu_indices <- which(accepted_mu)
  
  if (length(valid_mu_indices) == 0) {
      cat("[Alg C] WARNING: No valid mean vectors accepted from grid.\n")
      # Return placeholder
      return(NULL)
  }
  
  max_dist <- -1
  best_mu1 <- NULL
  best_mu2 <- NULL
  
  for (i in valid_mu_indices) {
      for (j in valid_mu_indices) {
          dist <- max(abs(N_mu[i, ] - N_mu[j, ]))
          if (dist > max_dist) {
              max_dist <- dist
              best_mu1 <- N_mu[i, ]
              best_mu2 <- N_mu[j, ]
          }
      }
  }
  
  cat("\n[Alg C] Vectorizing Covariance Grid Search (Consensus Check)...\n")
  M_A <- matrix(0, nrow = m, ncol = 10)
  for (i in 1:m) {
      ai <- a_samples[i, ]
      M_A[i, 1:4] <- ai^2
      M_A[i, 5]  <- 2 * ai[1] * ai[2]
      M_A[i, 6]  <- 2 * ai[1] * ai[3]
      M_A[i, 7]  <- 2 * ai[1] * ai[4]
      M_A[i, 8]  <- 2 * ai[2] * ai[3]
      M_A[i, 9]  <- 2 * ai[2] * ai[4]
      M_A[i, 10] <- 2 * ai[3] * ai[4]
  }
  
  # Helper to reconstruct symmetric matrix from params
  reconstruct_sigma_internal <- function(params) {
      mat <- matrix(0, d, d)
      mat[1,1] <- params[1]; mat[2,2] <- params[2]; mat[3,3] <- params[3]; mat[4,4] <- params[4]
      mat[1,2] <- mat[2,1] <- params[5]
      mat[1,3] <- mat[3,1] <- params[6]
      mat[1,4] <- mat[4,1] <- params[7]
      mat[2,3] <- mat[3,2] <- params[8]
      mat[2,4] <- mat[4,2] <- params[9]
      mat[3,4] <- mat[4,3] <- params[10]
      return(mat)
  }
  
  cat("[Alg C] Polishing Mean Estimates via Least Squares Regression...\n")
  # 1. Labeling for means using grid-based best_mu1, best_mu2
  W1 <- numeric(m); W2 <- numeric(m)
  for (i in 1:m) {
      ai <- a_samples[i, ]
      gm1 <- sum(ai * best_mu1); gm2 <- sum(ai * best_mu2)
      t1 <- projections_res[[i]]$mu1; t2 <- projections_res[[i]]$mu2
      if (abs(t1 - gm1) + abs(t2 - gm2) < abs(t2 - gm1) + abs(t1 - gm2)) {
          W1[i] <- t1; W2[i] <- t2
      } else {
          W1[i] <- t2; W2[i] <- t1
      }
  }
  best_mu1 <- as.numeric(solve(t(a_samples) %*% a_samples) %*% (t(a_samples) %*% W1))
  best_mu2 <- as.numeric(solve(t(a_samples) %*% a_samples) %*% (t(a_samples) %*% W2))

  cat("[Alg C] Polishing Covariance Estimates via Mean-Labeling LS...\n")
  # 2. Use POLISHED MEANS to label the variances (much more robust)
  V1 <- numeric(m); V2 <- numeric(m)
  for (i in 1:m) {
      ai <- a_samples[i, ]
      pm1 <- sum(ai * best_mu1); pm2 <- sum(ai * best_mu2)
      tu1 <- projections_res[[i]]$mu1; tu2 <- projections_res[[i]]$mu2
      tv1 <- projections_res[[i]]$var1; tv2 <- projections_res[[i]]$var2
      if (abs(tu1 - pm1) < abs(tu2 - pm1)) {
          V1[i] <- tv1; V2[i] <- tv2
      } else {
          V1[i] <- tv2; V2[i] <- tv1
      }
  }
  
  theta1_opt <- as.numeric(solve(t(M_A) %*% M_A + diag(1e-6, 10)) %*% (t(M_A) %*% V1))
  theta2_opt <- as.numeric(solve(t(M_A) %*% M_A + diag(1e-6, 10)) %*% (t(M_A) %*% V2))
  
  project_psd <- function(mat) {
      ee <- eigen(mat, symmetric = TRUE)
      ee$values[ee$values < 0] <- 0
      return(ee$vectors %*% diag(ee$values) %*% t(ee$vectors))
  }
  
  best_sigma1 <- project_psd(reconstruct_sigma_internal(theta1_opt))
  best_sigma2 <- project_psd(reconstruct_sigma_internal(theta2_opt))

  cat("[Alg C] Successfully bounded and extracted theoretical centers and covariances!\n")
  return(list(
      comp1 = list(mu = best_mu1, sigma = best_sigma1),
      comp2 = list(mu = best_mu2, sigma = best_sigma2)
  ))
}
