source("code_r/hardt_price_gmm_nd.R")

test_nd_variance <- function() {
  n <- 10000
  d <- 10
  p <- 0.3 # Component 1 probability
  
  # Means
  mu1 <- rep(0, d)
  mu2 <- rep(0, d)
  mu2[1] <- 5 # Dimension 1 has large mean separation
  
  # Covariance matrices (diagonal for simplicity)
  sigma1 <- rep(1, d)
  sigma2 <- rep(1, d)
  sigma2[2] <- 10 # Dimension 2 has large variance in component 2
  
  labels <- rbinom(n, 1, 1 - p)
  X <- matrix(0, nrow = n, ncol = d)
  
  idx1 <- which(labels == 0)
  idx2 <- which(labels == 1)
  
  for (j in 1:d) {
    if (length(idx1) > 0) X[idx1, j] <- rnorm(length(idx1), mu1[j], sqrt(sigma1[j]))
    if (length(idx2) > 0) X[idx2, j] <- rnorm(length(idx2), mu2[j], sqrt(sigma2[j]))
  }
  
  # True theoretical coordinate variances
  # Dimension 1 variance: p*(1-p)*(5-0)^2 + p*1 + (1-p)*1 = 0.21 * 25 + 1 = 6.25
  # Dimension 2 variance: 0 + p*1 + (1-p)*10 = 0.3*1 + 0.7*10 = 7.3
  # Expected max variance is around 7.3
  
  cat("\n=== Testing N-Dimensional Empirical Variance Estimator ===\n")
  cat("Testing on d=10 with hidden large mean gap in dim 1, large variance in dim 2\n")
  
  est_var <- estimate_d_dim_variance(X, delta = 0.05)
  
  cat(sprintf("\nEstimated Max Coordinate Variance: %.4f\n", est_var))
  cat("Expected: ~7.3\n")
  
  if (abs(est_var - 7.3) < 0.5) {
      cat("Status: PASS - Constant-factor Variance Successfully Recovered!\n")
  } else {
      cat("Status: FAIL\n")
  }
}

test_nd_variance()
