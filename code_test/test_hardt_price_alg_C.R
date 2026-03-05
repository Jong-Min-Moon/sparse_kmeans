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
