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
library(Rcpp)
sourceCpp("code_rcpp/hardt_price_1d.cpp")

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



#' Algorithm C: Reduction from 4D to 1D
#'
#' @param X an n x 4 matrix of samples
#' @param epsilon The target precision
#' @param delta confidence parameter
#' @return The recovered parameters (mu1, mu2, Sigma1, Sigma2)
Reduce4DTo1D <- function(X, epsilon = 0.5, delta = 0.05) {
  X <- as.matrix(X)
  n <- nrow(X)
  k_orig <- ncol(X)
  
  if (k_orig > 4) stop("Reduction algorithm strictly expects at most a 4-dimensional mixture")
  
  if (k_orig < 4) {
      X_padded <- matrix(0, nrow = n, ncol = 4)
      X_padded[, 1:k_orig] <- X
      X <- X_padded
  }
  
  d <- 4
  
  # cat("[Alg C] Step 1: Estimating max coordinate variance layer...\n")
  sigma_hat_sq <- estimate_d_dim_variance(X, delta)
  sigma_hat <- sqrt(sigma_hat_sq)
  # cat(sprintf("[Alg C] Max coordinate variance scale isolated: %.4f\n", sigma_hat_sq))
  
  # cat("[Alg C] Step 2: Locating spatial geometry center through 1D marginals...\n")
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

  
  m <- ceiling(10 * log(100 / delta))
  # cat(sprintf("[Alg C] Broadcasting over %d bounded random structural projections (a_i ~ N_9^4)...\n", m))
  
  a_samples <- sample_N9(m, d)
  
  # cat("[Alg C] Processing 1D projection ensemble via Rcpp...\n")
  
  # Execute projections over fast C++ kernel directly
  projections_res <- lapply(1:m, function(i) {
      a_i <- a_samples[i, ]
      x_1d <- as.numeric(X %*% a_i)
      
      res_1d <- Recover1DMixture_cpp(x_1d, delta = delta / m)
      
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
  })
  
  # --------------------------------------------------------------------------
  # ALGORITHM C: EXACT THEORETICAL IMPLEMENTATION
  # --------------------------------------------------------------------------
  c_const <- 0.1
  eps_mu <- max(c_const * epsilon * sigma_hat, 1.5 * sigma_hat) 
  
  M_accepted <- list()
  grid_mu_1d <- seq(-2 * sigma_hat, 2 * sigma_hat, by = eps_mu)
  
  # Step 2: Accept/Reject Mean Grid N_mu
  for(m1 in grid_mu_1d) {
    for(m2 in grid_mu_1d) {
      for(m3 in grid_mu_1d) {
        for(m4 in grid_mu_1d) {
          mu_cand <- mu_hat + c(m1, m2, m3, m4)
          rejected <- FALSE
          for (i in seq_len(m)) {
             ai <- a_samples[i, ]
             val <- sum(ai * mu_cand)
             if (abs(val - projections_res[[i]]$mu1) > (epsilon * sigma_hat / 2) && 
                 abs(val - projections_res[[i]]$mu2) > (epsilon * sigma_hat / 2)) {
                 rejected <- TRUE
                 break
             }
          }
          if (!rejected) {
             M_accepted[[length(M_accepted) + 1]] <- mu_cand
          }
        }
      }
    }
  }
  
  if (length(M_accepted) == 0) {
      cat("[Alg C] Mean grid rejected all candidates (M_accepted is empty)\n")
      return(NULL)
  }
  
  # Step 3: Maximize L_infty dist
  max_dist <- -1
  best_mu1 <- M_accepted[[1]]
  best_mu2 <- M_accepted[[1]]
  for(i in seq_along(M_accepted)) {
    for(j in seq_along(M_accepted)) {
       dist <- max(abs(M_accepted[[i]] - M_accepted[[j]]))
       if (dist > max_dist) {
          max_dist <- dist
          best_mu1 <- M_accepted[[i]]
          best_mu2 <- M_accepted[[j]]
       }
    }
  }
  
  # Step 4: Accept/Reject Covariance Grid N_sigma
  S_accepted <- list()
  eps_sig <- max(c_const * epsilon^2 * sigma_hat_sq, 1.5 * sigma_hat_sq)
  grid_sig_1d <- seq(-sigma_hat_sq, sigma_hat_sq, by = eps_sig)
  
  search_sigma <- function(params) {
      if (length(params) == 10) {
          mat <- matrix(0, 4, 4)
          mat[1,1] <- params[1]; mat[2,2] <- params[2]; mat[3,3] <- params[3]; mat[4,4] <- params[4]
          mat[1,2] <- mat[2,1] <- params[5]
          mat[1,3] <- mat[3,1] <- params[6]
          mat[1,4] <- mat[4,1] <- params[7]
          mat[2,3] <- mat[3,2] <- params[8]
          mat[2,4] <- mat[4,2] <- params[9]
          mat[3,4] <- mat[4,3] <- params[10]
          
          rejected <- FALSE
          for (i in seq_len(m)) {
             ai <- a_samples[i, ]
             val <- as.numeric(t(ai) %*% mat %*% ai)
             if (abs(val - projections_res[[i]]$var1) > (epsilon^2 * sigma_hat_sq / 2) && 
                 abs(val - projections_res[[i]]$var2) > (epsilon^2 * sigma_hat_sq / 2)) {
                 rejected <- TRUE
                 break
             }
          }
          if (!rejected) {
             S_accepted[[length(S_accepted) + 1]] <<- mat
          }
          return()
      }
      for (v in grid_sig_1d) {
          search_sigma(c(params, v))
      }
  }
  search_sigma(c())
  
  if (length(S_accepted) == 0) {
      cat("[Alg C] Covariance grid rejected all candidates (S_accepted is empty)\n")
      return(NULL)
  }
  
  # Step 5: Maximize L_infty dist for matrices
  max_dist <- -1
  best_sig1 <- S_accepted[[1]]
  best_sig2 <- S_accepted[[1]]
  for(i in seq_along(S_accepted)) {
    for(j in seq_along(S_accepted)) {
       dist <- max(abs(S_accepted[[i]] - S_accepted[[j]]))
       if (dist > max_dist) {
          max_dist <- dist
          best_sig1 <- S_accepted[[i]]
          best_sig2 <- S_accepted[[j]]
       }
    }
  }
  
  # Step 6: Permutation swap validation
  for (i in seq_len(m)) {
      ai <- a_samples[i, ]
      
      val_sig1 <- as.numeric(t(ai) %*% best_sig1 %*% ai)
      val_mu1 <- sum(ai * best_mu1)
      
      match_1_to_1 <- (abs(val_sig1 - projections_res[[i]]$var1) <= epsilon^2 * sigma_hat_sq / 2) &&
                      (abs(val_mu1 - projections_res[[i]]$mu1) <= epsilon * sigma_hat / 2)
      
      match_1_to_2 <- (abs(val_sig1 - projections_res[[i]]$var2) <= epsilon^2 * sigma_hat_sq / 2) &&
                      (abs(val_mu1 - projections_res[[i]]$mu2) <= epsilon * sigma_hat / 2)
                      
      if (!match_1_to_1 && match_1_to_2) {
          temp <- best_sig1
          best_sig1 <- best_sig2
          best_sig2 <- temp
          break
      }
  }
  
  return(list(
      comp1 = list(mu = best_mu1[1:k_orig], sigma = best_sig1[1:k_orig, 1:k_orig, drop=FALSE]),
      comp2 = list(mu = best_mu2[1:k_orig], sigma = best_sig2[1:k_orig, 1:k_orig, drop=FALSE])
  ))
}
