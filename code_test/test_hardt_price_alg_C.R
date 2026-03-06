source("code_r/hardt_price_alg_C.R")
source("code_r/hardt_price_gmm_1d.R")
source("code_r/hardt_price_gmm_nd.R")

run_alg_c_test <- function(test_name, n, p, mu1, mu2, Sigma1, Sigma2, epsilon=0.4, delta=0.05) {
  cat(sprintf("\n\n=======================================================\n"))
  cat(sprintf("=== %s ===\n", test_name))
  cat(sprintf("=======================================================\n"))
  
  d <- length(mu1)
  
  set.seed(123)
  labels <- rbinom(n, 1, 1 - p)
  X <- matrix(0, nrow = n, ncol = d)
  
  idx1 <- which(labels == 0)
  idx2 <- which(labels == 1)
  
  # Manual MVN sampling to avoid dependency on MASS for now
  sample_mvn <- function(n_samples, mu, Sigma) {
      if (n_samples == 0) return(matrix(0, 0, length(mu)))
      Z <- matrix(rnorm(n_samples * length(mu)), n_samples, length(mu))
      # Cholesky decomposition: Sigma = L L^T
      L <- chol(Sigma) # Note: R's chol returns upper triangular U, Sigma = U^T U
      return(t(apply(Z, 1, function(z) mu + as.numeric(t(L) %*% z))))
  }
  
  if (length(idx1) > 0) X[idx1, ] <- sample_mvn(length(idx1), mu1, Sigma1)
  if (length(idx2) > 0) X[idx2, ] <- sample_mvn(length(idx2), mu2, Sigma2)
  
  tryCatch({
      res <- Reduce4DTo1D(X, epsilon = epsilon, delta = delta)
      
      cat("\n=== Recovered Means ===\n")
      cat("Expected mu1:", round(mu1, 3), "\n")
      cat("Expected mu2:", round(mu2, 3), "\n\n")
      
      cat("Recovered A:", round(res$comp1$mu, 3), "\n")
      cat("Recovered B:", round(res$comp2$mu, 3), "\n")
      
      cat("\n=== Recovered Sigmas (Top-Left 2x2 Block) ===\n")
      cat("Expected S1[1:2,1:2]:\n"); print(round(Sigma1[1:2, 1:2], 3))
      cat("Expected S2[1:2,1:2]:\n"); print(round(Sigma2[1:2, 1:2], 3))

      if (!is.null(res$comp1$sigma)) {
          # Frobenius norm of error
          err_f_11 <- sqrt(sum((Sigma1 - res$comp1$sigma)^2))
          err_f_12 <- sqrt(sum((Sigma1 - res$comp2$sigma)^2))
          err_f_21 <- sqrt(sum((Sigma2 - res$comp1$sigma)^2))
          err_f_22 <- sqrt(sum((Sigma2 - res$comp2$sigma)^2))
          
          # Match pairs
          if ((err_f_11 + err_f_22) < (err_f_12 + err_f_21)) {
              cat("\nRecovered A Sigma[1:2,1:2]:\n"); print(round(res$comp1$sigma[1:2, 1:2], 3))
              cat("Recovered B Sigma[1:2,1:2]:\n"); print(round(res$comp2$sigma[1:2, 1:2], 3))
              cat(sprintf("\nFrobenius Norm Error A: %.4f\n", err_f_11))
              cat(sprintf("Frobenius Norm Error B: %.4f\n", err_f_22))
          } else {
              cat("\nRecovered A Sigma[1:2,1:2]:\n"); print(round(res$comp2$sigma[1:2, 1:2], 3))
              cat("Recovered B Sigma[1:2,1:2]:\n"); print(round(res$comp1$sigma[1:2, 1:2], 3))
              cat(sprintf("\nFrobenius Norm Error A: %.4f\n", err_f_21))
              cat(sprintf("Frobenius Norm Error B: %.4f\n", err_f_12))
          }
      }
      
      errA <- min(max(abs(mu1 - res$comp1$mu)), max(abs(mu2 - res$comp1$mu)))
      errB <- min(max(abs(mu1 - res$comp2$mu)), max(abs(mu2 - res$comp2$mu)))
      cat(sprintf("\nMax coordinate error bound (L_inf): %.4f\n", max(errA, errB)))
    
  }, error = function(e) {
      cat("Algorithm C execution failed:\n")
      print(e$message)
  })
}

n <- 1000000
d <- 4

# Instance 1: Imbalanced Weights
run_alg_c_test(
  test_name = "Instance 1: Imbalanced Weights (p=0.3)",
  n = n, p = 0.3, 
  mu1 = c(-1.5, 0.5, 1.5, -0.5), 
  mu2 = c(1.5, -0.5, -1.5, 0.5),
  Sigma1 = diag(1, d), Sigma2 = diag(1, d),
  epsilon = 0.4
)

# Instance 2: Tighter Geometric Geometry
run_alg_c_test(
  test_name = "Instance 2: Tighter Geometric Geometry",
  n = n, p = 0.5, 
  mu1 = c(-1, -1, 1, 1), 
  mu2 = c(1, 1, -1, -1),
  Sigma1 = diag(1, d), Sigma2 = diag(1, d),
  epsilon = 0.4
)

# Instance 3: Different variances across dimensions
run_alg_c_test(
  test_name = "Instance 3: Heterogeneous Dimensional Variances",
  n = n, p = 0.4, 
  mu1 = c(-2.0, 0, 2.0, 0), 
  mu2 = c(2.0, 0, -2.0, 0),
  Sigma1 = diag(c(0.5, 1, 1.5, 2), d), Sigma2 = diag(c(0.5, 1, 1.5, 2), d),
  epsilon = 0.4
)

# Instance 4: Correlated Covariances (Off-diagonal recovery)
S4_1 <- matrix(c(1.0, 0.5, 0.3, 0.1,
                 0.5, 1.0, 0.2, 0.1,
                 0.3, 0.2, 1.0, 0.4,
                 0.1, 0.1, 0.4, 1.0), 4, 4)
S4_2 <- matrix(c(1.5, -0.4, 0.1, 0.0,
                 -0.4, 0.8, 0.2, 0.1,
                 0.1, 0.2, 1.2, -0.3,
                 0.0, 0.1, -0.3, 0.9), 4, 4)
run_alg_c_test(
  test_name = "Instance 4: Correlated Covariances (Off-diagonal recovery)",
  n = n, p = 0.5,
  mu1 = c(-2, -2, 2, 2),
  mu2 = c(2, 2, -2, -2),
  Sigma1 = S4_1, Sigma2 = S4_2
)

# Instance 5: Rotated Components
# Comp 1: Stretched in X+Y, Comp 2: Stretched in X-Y
run_alg_c_test(
  test_name = "Instance 5: Strong Rotation / Asymmetric Correlation",
  n = n, p = 0.5,
  mu1 = c(0, 0, 0, 0),
  mu2 = c(3, 3, 3, 3),
  Sigma1 = matrix(c(2, 1.5, 0, 0, 1.5, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), 4, 4),
  Sigma2 = matrix(c(2, -1.5, 0, 0, -1.5, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), 4, 4)
)
