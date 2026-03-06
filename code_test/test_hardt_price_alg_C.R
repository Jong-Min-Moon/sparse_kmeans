source("code_r/hardt_price_alg_C.R")
source("code_r/hardt_price_gmm_1d.R")
source("code_r/hardt_price_gmm_nd.R")

run_alg_c_test <- function(test_name, n, p, mu1, mu2, sigma1, sigma2, epsilon=0.4, delta=0.05) {
  cat(sprintf("\n\n=======================================================\n"))
  cat(sprintf("=== %s ===\n", test_name))
  cat(sprintf("=======================================================\n"))
  
  d <- length(mu1)
  
  set.seed(123)
  labels <- rbinom(n, 1, 1 - p)
  X <- matrix(0, nrow = n, ncol = d)
  
  idx1 <- which(labels == 0)
  idx2 <- which(labels == 1)
  
  if (length(idx1) > 0) {
      for (j in 1:d) X[idx1, j] <- rnorm(length(idx1), mu1[j], sqrt(sigma1[j]))
  }
  if (length(idx2) > 0) {
      for (j in 1:d) X[idx2, j] <- rnorm(length(idx2), mu2[j], sqrt(sigma2[j]))
  }
  
  tryCatch({
      res <- Reduce4DTo1D(X, epsilon = epsilon, delta = delta)
      
      cat("\n=== Recovered Means ===\n")
      cat("Expected mu1:", round(mu1, 3), "\n")
      cat("Expected mu2:", round(mu2, 3), "\n\n")
      
      cat("Recovered A:", round(res$comp1$mu, 3), "\n")
      cat("Recovered B:", round(res$comp2$mu, 3), "\n")
      
      cat("\n=== Recovered Sigmas (Diagonals) ===\n")
      cat("Expected sigma1:", round(sigma1, 3), "\n")
      cat("Expected sigma2:", round(sigma2, 3), "\n\n")

      if (!is.null(res$comp1$sigma)) {
          cat("Recovered A Sigma Diag:", round(diag(res$comp1$sigma), 3), "\n")
          cat("Recovered B Sigma Diag:", round(diag(res$comp2$sigma), 3), "\n")
          
          # Frobenius norm of error
          S1 <- diag(sigma1, d)
          S2 <- diag(sigma2, d)
          
          err_f_11 <- sqrt(sum((S1 - res$comp1$sigma)^2))
          err_f_12 <- sqrt(sum((S1 - res$comp2$sigma)^2))
          err_f_21 <- sqrt(sum((S2 - res$comp1$sigma)^2))
          err_f_22 <- sqrt(sum((S2 - res$comp2$sigma)^2))
          
          # Match pairs
          if ((err_f_11 + err_f_22) < (err_f_12 + err_f_21)) {
              cat(sprintf("\nFrobenius Norm Error A: %.4f\n", err_f_11))
              cat(sprintf("Frobenius Norm Error B: %.4f\n", err_f_22))
          } else {
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
  sigma1 = rep(1, d), sigma2 = rep(1, d),
  epsilon = 0.4
)

# Instance 2: Tighter Geometric Geometry
run_alg_c_test(
  test_name = "Instance 2: Tighter Geometric Geometry",
  n = n, p = 0.5, 
  mu1 = c(-1, -1, 1, 1), 
  mu2 = c(1, 1, -1, -1),
  sigma1 = rep(1, d), sigma2 = rep(1, d),
  epsilon = 0.4
)

# Instance 3: Different variances across dimensions
run_alg_c_test(
  test_name = "Instance 3: Heterogeneous Dimensional Variances",
  n = n, p = 0.4, 
  mu1 = c(-2.0, 0, 2.0, 0), 
  mu2 = c(2.0, 0, -2.0, 0),
  sigma1 = c(0.5, 1, 1.5, 2), sigma2 = c(0.5, 1, 1.5, 2),
  epsilon = 0.4
)
