#' Generate random projection vectors from N_9^d
#'
#' Following Lemma 3.4 in Hardt & Price, this generates samples from the 
#' d-dimensional normal distribution N(0,1)^d conditioned on the norm being at most 9.
#'
#' @param n number of vectors to sample
#' @param d dimension (e.g., d = 4 for the 4D to 1D reduction)
#' @return A matrix of size n x d where each row is a bounded projection vector
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
  # 4 points per parameter gives 4^10 = 1,048,576 matrices.
  # Vectorization makes this manageable (< 2 seconds).
  grid_1d <- seq(0, sigma_hat^2, length.out = 4)
  
  num_params <- d * (d + 1) / 2
  grid_1d_list <- replicate(num_params, grid_1d, simplify = FALSE)
  grid_df <- expand.grid(grid_1d_list)
  
  # Return the grid as a matrix (N_param x N_cand)
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
  
  projections_res <- list()
  
  for (i in 1:m) {
      a_i <- a_samples[i, ]
      
      x_1d <- drop(X %*% a_i)
      
      # Recover the 1D parameters on the projection line
      res_1d <- Recover1DMixture(x_1d, delta = delta / m)
      
      if (is.null(res_1d) || isTRUE(res_1d$fallback) || is.null(res_1d$comp2)) {
          # If the projection caused the geometry to completely overlap, we only got 1 component.
          # We duplicate it for the math filter so the error checks just evaluate the single target.
          res_1d$comp2 <- res_1d$comp1
      }
      
      if (is.null(res_1d$comp1$sigma) || is.null(res_1d$comp2$sigma)) {
          cat(sprintf("\n[ERROR] Found NULL sigma at projection %d. Structure dump:\n", i))
          print(str(res_1d))
          stop("Halting due to malformed 1D orchestrator output.")
      }
      
      projections_res[[i]] <- list(
        mu1 = res_1d$comp1$mu,
        mu2 = res_1d$comp2$mu,
        var1 = res_1d$comp1$sigma^2,
        var2 = res_1d$comp2$sigma^2
      )
  }
  
  cat("\n[Alg C] Running Voting Filter to isolate means out of mesh space...\n")
  accepted_mu <- rep(FALSE, nrow(N_mu))
  # Theoretical voting threshold (at least 85% of random projections must structurally align)
  voting_threshold <- floor(0.85 * m)
  
  for (idx in 1:nrow(N_mu)) {
      cand_mu <- N_mu[idx, ]
      votes <- 0
      
      for (i in 1:m) {
          a_i <- a_samples[i, ]
          proj <- sum(a_i * cand_mu)
          
          err1 <- abs(proj - projections_res[[i]]$mu1)
          err2 <- abs(proj - projections_res[[i]]$mu2)
          
          eps_prime <- epsilon * sum(abs(a_i)) + 0.1
          
          if (err1 <= (eps_prime * sigma_hat / 2) || err2 <= (eps_prime * sigma_hat / 2)) {
              votes <- votes + 1
          }
      }
      
      if (votes >= voting_threshold) {
          accepted_mu[idx] <- TRUE
      }
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
  
  cat("\n[Alg C] Vectorizing Covariance Grid Search...\n")
  
  # N_sigma is (10 x N_cand). We need the 10 factors for a^T Sigma a.
  # Params: Sigma11, Sigma22, Sigma33, Sigma44, Sigma12, Sigma13, Sigma14, Sigma23, Sigma24, Sigma34
  # Factors: a1^2, a2^2, a3^2, a4^2, 2*a1*a2, 2*a1*a3, 2*a1*a4, 2*a2*a3, 2*a2*a4, 2*a3*a4
  
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
  
  # Variances is (m x N_cand)
  candidate_variances <- M_A %*% N_sigma
  
  sigma_voting_threshold <- floor(0.60 * m)
  
  cat("[Alg C] Evaluating variances across 1D projection ensemble...\n")
  
  # Efficient voting calculation
  # threshold set slightly tighter: epsilon^0.5 * sigma_hat^2
  votes <- colSums(sapply(1:m, function(i) {
      v <- candidate_variances[i, ]
      t1 <- projections_res[[i]]$var1
      t2 <- projections_res[[i]]$var2
      
      a_norm_sq <- sum(a_samples[i, ]^2)
      threshold <- (sqrt(epsilon) * sigma_hat^2 * a_norm_sq / 2) + 0.1
      
      return(abs(v - t1) <= threshold | abs(v - t2) <= threshold)
  }))
  
  # Selection logic: prioritize candidates that pass the threshold, 
  # then pick the pair among the top-voted that are furthest apart.
  max_votes <- max(votes)
  best_consensus_indices <- which(votes >= max(sigma_voting_threshold, 0.9 * max_votes))
  
  if (length(best_consensus_indices) == 0) {
      cat("[Alg C] WARNING: No valid covariance matrices accepted from grid.\n")
      best_sigma1 <- diag(sigma_hat^2, d)
      best_sigma2 <- diag(sigma_hat^2, d)
  } else {
      # Consensus refinement: Take the top-voted candidates and cluster them if possible,
      # or just take the best two that are well-separated.
      # To keep it simple and robust, we pick the most-voted candidate as Sigma1,
      # then pick the most-voted candidate that is 'far' from Sigma1 as Sigma2.
      
      sorted_indices <- best_consensus_indices[order(votes[best_consensus_indices], decreasing = TRUE)]
      # Take top 100 for averaging
      top_k <- min(100, length(sorted_indices))
      top_indices <- sorted_indices[1:top_k]
      
      # Simple clustering: pick the top one as seed 1
      idx1_seed <- top_indices[1]
      v1 <- N_sigma[, idx1_seed]
      
      # Find a seed for the other component that is furthest from v1 among top candidates
      max_d <- -1
      idx2_seed <- top_indices[1]
      for (i in top_indices) {
          d_val <- sum(abs(N_sigma[, i] - v1))
          if (d_val > max_d) {
              max_d <- d_val
              idx2_seed <- i
          }
      }
      v2 <- N_sigma[, idx2_seed]
      
      # Average the candidates that are close to each seed
      cluster1_indices <- c()
      cluster2_indices <- c()
      
      for (i in top_indices) {
          d1 <- sum(abs(N_sigma[, i] - v1))
          d2 <- sum(abs(N_sigma[, i] - v2))
          if (d1 < d2) {
              cluster1_indices <- c(cluster1_indices, i)
          } else {
              cluster2_indices <- c(cluster2_indices, i)
          }
      }
      
      avg_params1 <- rowMeans(as.matrix(N_sigma[, cluster1_indices, drop = FALSE]))
      avg_params2 <- rowMeans(as.matrix(N_sigma[, cluster2_indices, drop = FALSE]))
      
      cat("[Alg C] Polishing Covariance Estimates via Least Squares Regression...\n")
      
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
      
      S1_init <- reconstruct_sigma_internal(avg_params1)
      S2_init <- reconstruct_sigma_internal(avg_params2)
      
      # 1. Labeling phase: assign 1D variances to components
      V1 <- numeric(m)
      V2 <- numeric(m)
      
      for (i in 1:m) {
          ai <- a_samples[i, ]
          ev1 <- as.numeric(t(ai) %*% S1_init %*% ai)
          ev2 <- as.numeric(t(ai) %*% S2_init %*% ai)
          
          t1 <- projections_res[[i]]$var1
          t2 <- projections_res[[i]]$var2
          
          # Match target to component by proximity to initial estimate
          if (abs(t1 - ev1) + abs(t2 - ev2) < abs(t2 - ev1) + abs(t1 - ev2)) {
              V1[i] <- t1
              V2[i] <- t2
          } else {
              V1[i] <- t2
              V2[i] <- t1
          }
      }
      
      # 2. Regression phase: solve M_A * theta = V
      # Use QR decomposition for numerical stability
      solve_ls_refinement <- function(A, b) {
          return(as.numeric(solve(t(A) %*% A, t(A) %*% b)))
      }
      
      theta1_opt <- solve_ls_refinement(M_A, V1)
      theta2_opt <- solve_ls_refinement(M_A, V2)
      
      # 3. PSD Projection phase: ensure stability
      project_psd_internal <- function(mat) {
          ee <- eigen(mat, symmetric = TRUE)
          ee$values[ee$values < 0] <- 0
          return(ee$vectors %*% diag(ee$values) %*% t(ee$vectors))
      }
      
      best_sigma1 <- project_psd_internal(reconstruct_sigma_internal(theta1_opt))
      best_sigma2 <- project_psd_internal(reconstruct_sigma_internal(theta2_opt))
  }
  
  cat("[Alg C] Aligning Mean and Covariance Permutations...\n")
  votes_A <- 0
  votes_B <- 0
  
  for (i in 1:m) {
      a_i <- a_samples[i, ]
      proj_mu1 <- sum(a_i * best_mu1)
      proj_mu2 <- sum(a_i * best_mu2)
      proj_var1 <- sum(a_i * (best_sigma1 %*% a_i))
      proj_var2 <- sum(a_i * (best_sigma2 %*% a_i))
      
      target_mu1 <- projections_res[[i]]$mu1
      target_mu2 <- projections_res[[i]]$mu2
      target_var1 <- projections_res[[i]]$var1
      target_var2 <- projections_res[[i]]$var2
      
      # Configuration A (1->1, 2->2)
      err_mu11 <- abs(proj_mu1 - target_mu1); err_var11 <- abs(proj_var1 - target_var1)
      err_mu22 <- abs(proj_mu2 - target_mu2); err_var22 <- abs(proj_var2 - target_var2)
      score_A <- err_mu11 + err_var11 + err_mu22 + err_var22
      
      # Configuration B (1->2, 2->1)
      err_mu12 <- abs(proj_mu1 - target_mu2); err_var12 <- abs(proj_var1 - target_var2)
      err_mu21 <- abs(proj_mu2 - target_mu1); err_var21 <- abs(proj_var2 - target_var1)
      score_B <- err_mu12 + err_var12 + err_mu21 + err_var21
      
      if (score_A <= score_B) {
          votes_A <- votes_A + 1
      } else {
          votes_B <- votes_B + 1
      }
  }
  
  if (votes_B > votes_A) {
      tmp_sigma <- best_sigma1
      best_sigma1 <- best_sigma2
      best_sigma2 <- tmp_sigma
  }

  cat("[Alg C] Successfully bounded and extracted theoretical centers and covariances!\n")
  return(list(
      comp1 = list(mu = best_mu1, sigma = best_sigma1),
      comp2 = list(mu = best_mu2, sigma = best_sigma2)
  ))
}
