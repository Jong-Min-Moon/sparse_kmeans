source("code_r/hardt_price_alg_C.R")

#' Algorithm B: Reduction from d dimensions to 4 dimensions
#'
#' @param X an n x d matrix of samples
#' @param epsilon The target precision
#' @param delta confidence parameter
#' @return The recovered parameters (mu1, mu2, Sigma1, Sigma2)
ReduceDTo4 <- function(X, epsilon = 0.5, delta = 0.05) {
  X <- as.matrix(X)
  n <- nrow(X)
  d <- ncol(X)
  
  if (d <= 4) {
      cat(sprintf("[Alg B] Dimension d=%d <= 4. Deferring entirely to 4D oracle.\n", d))
      return(Reduce4DTo1D(X, epsilon, delta))
  }
  
  cat(sprintf("[Alg B] Extending 4D Oracle to %d Dimensions...\n", d))
  
  sigma_hat_sq <- estimate_d_dim_variance(X, delta)
  sigma_hat <- sqrt(sigma_hat_sq)
  
  # Shared confidence for sub-calls
  delta_prime <- delta / (10 * d^2)
  eps_prime <- epsilon / 20
  
  # ---------------------------------------------------------------------
  # STEP 1: Mean Recovery
  # ---------------------------------------------------------------------
  cat("[Alg B] Anchoring Means via 1D/2D Projections...\n")
  mu_hat_1 <- numeric(d)
  mu_hat_2 <- numeric(d)
  
  xi_means <- list()
  for (i in 1:d) {
      res_1d <- Reduce4DTo1D(X[, i, drop=FALSE], epsilon = eps_prime, delta = delta_prime)
      if (is.null(res_1d)) {
          # Handle catastrophic failure by falling back to empirical mean
          m_emp <- mean(X[, i])
          xi_means[[i]] <- c(m_emp, m_emp)
      } else {
          xi_means[[i]] <- c(res_1d$comp1$mu[1], res_1d$comp2$mu[1])
      }
  }
  
  cat("[Alg B Diagnostics] xi_means[[1]] evaluated as:\n")
  print(xi_means[[1]])
  
  mean_anchor_idx <- -1
  mean_thresh <- (eps_prime * sigma_hat) / 4
  
  for (i in 1:d) {
      if (abs(xi_means[[i]][1] - xi_means[[i]][2]) > mean_thresh) {
          mean_anchor_idx <- i
          break
      }
  }
  
  if (mean_anchor_idx == -1) {
      cat("[Alg B] Means are indistinguishable. Assigning identical mean components.\n")
      for (i in 1:d) {
          mu_hat_1[i] <- xi_means[[i]][1]
          mu_hat_2[i] <- xi_means[[i]][1]
      }
  } else {
      cat(sprintf("[Alg B] Found strong mean anchor at coordinate %d.\n", mean_anchor_idx))
      i <- mean_anchor_idx
      mu_hat_1[i] <- xi_means[[i]][1]
      mu_hat_2[i] <- xi_means[[i]][2]
      
      for (j in setdiff(1:d, i)) {
          # Query 2D oracle on (i, j)
          idx_query <- c(i, j)
          res_2d <- Reduce4DTo1D(X[, idx_query, drop=FALSE], epsilon = eps_prime, delta = delta_prime)
          
          if (is.null(res_2d)) {
              m_emp <- mean(X[, j])
              mu_hat_1[j] <- m_emp; mu_hat_2[j] <- m_emp
              next
          }
          
          nu_i_1 <- res_2d$comp1$mu[1]
          nu_i_2 <- res_2d$comp2$mu[1]
          nu_j_1 <- res_2d$comp1$mu[2]
          nu_j_2 <- res_2d$comp2$mu[2]
          
          # Match to anchor
          dist1 <- abs(xi_means[[i]][1] - nu_i_1)
          dist2 <- abs(xi_means[[i]][1] - nu_i_2)
          
          tau_mu <- (epsilon * sigma_hat) / 10
          if (dist1 > tau_mu && dist2 > tau_mu) {
              return(NULL) # "failure"
          }
          
          if (dist1 < dist2) {
              mu_hat_1[j] <- nu_j_1
              mu_hat_2[j] <- nu_j_2
          } else {
              mu_hat_1[j] <- nu_j_2
              mu_hat_2[j] <- nu_j_1
          }
      }
  }
  
  # ---------------------------------------------------------------------
  # STEP 2: Covariance Recovery
  # ---------------------------------------------------------------------
  cat("[Alg B] Anchoring Covariances via 2D/4D Projections...\n")
  Sigma_hat_1 <- matrix(0, d, d)
  Sigma_hat_2 <- matrix(0, d, d)
  
  cov_anchor_i <- -1
  cov_anchor_j <- -1
  cov_thresh <- (eps_prime^2 * sigma_hat_sq) / 4
  
  xi_cov <- matrix(list(), d, d)
  
  # Search for a covariance anchor by iteratively evaluating 2D bounds
  for(i in 1:d) {
      if (cov_anchor_i != -1) break
      for(j in i:d) {
          if (cov_anchor_i != -1) break
          
          idx_query <- unique(c(i, j))
          res_2d <- Reduce4DTo1D(X[, idx_query, drop=FALSE], epsilon = eps_prime, delta = delta_prime)
          
          if (is.null(res_2d)) {
              v_emp <- cov(X[, idx_query, drop=FALSE])
              v1 <- v_emp[1, length(idx_query)]; v2 <- v_emp[1, length(idx_query)]
          } else {
              if (length(idx_query) == 1) {
                  v1 <- res_2d$comp1$sigma[1, 1]
                  v2 <- res_2d$comp2$sigma[1, 1]
              } else {
                  v1 <- res_2d$comp1$sigma[1, 2]
                  v2 <- res_2d$comp2$sigma[1, 2]
              }
          }
          
          xi_cov[[i, j]] <- c(v1, v2)
          
          if (abs(v1 - v2) > cov_thresh) {
              cov_anchor_i <- i
              cov_anchor_j <- j
              break
          }
      }
  }
  
  if (cov_anchor_i == -1) {
      cat("[Alg B] Covariance entries are indistinguishable. Assigning identically.\n")
      cat("[Alg B] Covariance entries are indistinguishable. Assigning from 2D oracles.\n")
      for (i in 1:d) {
          for (j in i:d) {
              Sigma_hat_1[i, j] <- Sigma_hat_1[j, i] <- xi_cov[[i, j]][1]
              Sigma_hat_2[i, j] <- Sigma_hat_2[j, i] <- xi_cov[[i, j]][1]
          }
      }
  } else {
      cat(sprintf("[Alg B] Found strong covariance anchor at pair (%d, %d).\n", cov_anchor_i, cov_anchor_j))
      anchor_val_1 <- xi_cov[[cov_anchor_i, cov_anchor_j]][1]
      
      # Fill anchor
      Sigma_hat_1[cov_anchor_i, cov_anchor_j] <- Sigma_hat_1[cov_anchor_j, cov_anchor_i] <- xi_cov[[cov_anchor_i, cov_anchor_j]][1]
      Sigma_hat_2[cov_anchor_i, cov_anchor_j] <- Sigma_hat_2[cov_anchor_j, cov_anchor_i] <- xi_cov[[cov_anchor_i, cov_anchor_j]][2]
      
      # To handle D=500 efficiently, we parallelize the 125,000 independent 4D oracle checks.
      # We construct a list of all query targets, execute in parallel, and re-assign.
      cat(sprintf("[Alg B] Dispatching %d independent 4D oracle queries over parallel workers...\n", d * (d + 1) / 2 - 1))
      
      queries <- list()
      q_idx <- 1
      for (k in 1:d) {
          for (l in k:d) {
              if (k == cov_anchor_i && l == cov_anchor_j) next
              queries[[q_idx]] <- c(k, l)
              q_idx <- q_idx + 1
          }
      }
      
      # Execute via multi-core fork to leverage native macOS parallelization with shared C++ pointers
      library(parallel)
      n_cores <- max(1, detectCores() - 1)
      results_list <- mclapply(seq_along(queries), function(idx) {
          if (idx %% 5000 == 0) cat(sprintf("[Alg B] Processed %d / %d queries...\n", idx, length(queries)))
          q <- queries[[idx]]
          k <- q[1]
          l <- q[2]
          
          idx_query <- c(cov_anchor_i, cov_anchor_j, k, l)
          u_idx <- unique(idx_query)
          
          pos_i <- match(cov_anchor_i, u_idx)
          pos_j <- match(cov_anchor_j, u_idx)
          pos_k <- match(k, u_idx)
          pos_l <- match(l, u_idx)
          
          res_4d <- Reduce4DTo1D(X[, u_idx, drop=FALSE], epsilon = eps_prime, delta = delta_prime)
          
          if (is.null(res_4d)) {
              v_emp <- cov(X[, u_idx, drop=FALSE])
              v1 <- v_emp[pos_k, pos_l]; v2 <- v_emp[pos_k, pos_l]
          } else {
              sigma_ij_1 <- res_4d$comp1$sigma[pos_i, pos_j]
              sigma_ij_2 <- res_4d$comp2$sigma[pos_i, pos_j]
              sigma_kl_1 <- res_4d$comp1$sigma[pos_k, pos_l]
              sigma_kl_2 <- res_4d$comp2$sigma[pos_k, pos_l]
              
              dist1 <- abs(anchor_val_1 - sigma_ij_1)
              dist2 <- abs(anchor_val_1 - sigma_ij_2)
              
              tau_sig <- (epsilon^2 * sigma_hat_sq) / 10
              if (dist1 > tau_sig && dist2 > tau_sig) {
                  return(list(k=k, l=l, v1=NULL, v2=NULL, failed=TRUE))
              }
              
              if (dist1 < dist2) {
                  v1 <- sigma_kl_1; v2 <- sigma_kl_2
              } else {
                  v1 <- sigma_kl_2; v2 <- sigma_kl_1
              }
          }
          return(list(k=k, l=l, v1=v1, v2=v2))
      }, mc.cores = n_cores)
      
      for (res in results_list) {
          if (isTRUE(res$failed)) return(NULL) # "failure"
          Sigma_hat_1[res$k, res$l] <- Sigma_hat_1[res$l, res$k] <- res$v1
          Sigma_hat_2[res$k, res$l] <- Sigma_hat_2[res$l, res$k] <- res$v2
      }
  }
  
  
  # ---------------------------------------------------------------------
  # STEP 3: Matching up Sigma and mu combinations
  # ---------------------------------------------------------------------
  if (mean_anchor_idx != -1 && cov_anchor_i != -1) {
      cat("[Alg B] Global sign alignment between independent Means and Covariances...\n")
      
      idx_query <- unique(c(cov_anchor_i, cov_anchor_j, mean_anchor_idx))
      pos_i <- match(cov_anchor_i, idx_query)
      pos_j <- match(cov_anchor_j, idx_query)
      pos_k <- match(mean_anchor_idx, idx_query)
      
      res_3d <- Reduce4DTo1D(X[, idx_query, drop=FALSE], epsilon = eps_prime, delta = delta_prime)
      
      if (!is.null(res_3d)) {
          # Compare parings in res_3d to our assembled components.
          # From our assembled components, Comp 1 expects:
          expected_mu_1 <- mu_hat_1[mean_anchor_idx]
          expected_sig_1 <- Sigma_hat_1[cov_anchor_i, cov_anchor_j]
          
          # From 3D oracle:
          oracle_mu_1 <- res_3d$comp1$mu[pos_k]
          oracle_sig_1 <- res_3d$comp1$sigma[pos_i, pos_j]
          
          oracle_mu_2 <- res_3d$comp2$mu[pos_k]
          oracle_sig_2 <- res_3d$comp2$sigma[pos_i, pos_j]
          
          # Which oracle component matches our assembled mu_1?
          dist_mu1 <- abs(expected_mu_1 - oracle_mu_1)
          dist_mu2 <- abs(expected_mu_1 - oracle_mu_2)
          
          # Which oracle component matches our assembled sig_1?
          dist_sig1 <- abs(expected_sig_1 - oracle_sig_1)
          dist_sig2 <- abs(expected_sig_1 - oracle_sig_2)
          
          # We need to see if the assembled pair (mu_1, sig_1) came from the SAME underlying component
          matched_mu_idx <- if (dist_mu1 < dist_mu2) 1 else 2
          matched_sig_idx <- if (dist_sig1 < dist_sig2) 1 else 2
          
          if (matched_mu_idx != matched_sig_idx) {
              cat("[Alg B] Unaligned signatures detected! Swapping covariance assignments.\n")
              temp <- Sigma_hat_1
              Sigma_hat_1 <- Sigma_hat_2
              Sigma_hat_2 <- temp
          }
      }
  }
  
  return(list(
      comp1 = list(mu = mu_hat_1, sigma = Sigma_hat_1),
      comp2 = list(mu = mu_hat_2, sigma = Sigma_hat_2)
  ))
}
