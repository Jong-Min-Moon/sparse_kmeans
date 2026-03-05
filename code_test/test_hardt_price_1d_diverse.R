source("code_r/hardt_price_gmm_1d.R")

generate_gmm_data <- function(n, p, mu1, mu2, sigma1, sigma2) {
  # Generate labels
  labels <- rbinom(n, 1, 1 - p) # 0 maps to comp1, 1 to comp2
  x <- numeric(n)
  
  idx1 <- which(labels == 0)
  idx2 <- which(labels == 1)
  
  if (length(idx1) > 0) x[idx1] <- rnorm(length(idx1), mu1, sigma1)
  if (length(idx2) > 0) x[idx2] <- rnorm(length(idx2), mu2, sigma2)
  
  return(list(x = x, labels = labels))
}

test_scenario <- function(name, n, p, mu1, mu2, sigma1, sigma2, delta = 0.05) {
  cat(sprintf("\n--- Scenario: %s ---\n", name))
  cat(sprintf("True Parameters:\n"))
  cat(sprintf("Comp 1: p = %.2f, mu = %.2f, sigma = %.2f\n", p, mu1, sigma1))
  cat(sprintf("Comp 2: p = %.2f, mu = %.2f, sigma = %.2f\n", 1-p, mu2, sigma2))
  
  data <- generate_gmm_data(n, p, mu1, mu2, sigma1, sigma2)
  
  res <- Recover1DMixture(data$x, delta = delta)
  
  cat(sprintf("Recovered Parameters:\n"))
  if (isTRUE(res$fallback)) {
      cat("Output: FALLBACK (Single Component)\n")
      cat(sprintf("Comp: p = %.2f, mu = %.2f, sigma = %.2f\n", res$comp1$p, res$comp1$mu, res$comp1$sigma))
  } else {
      cat(sprintf("Comp A: p = %.2f, mu = %.2f, sigma = %.2f\n", res$comp1$p, res$comp1$mu, res$comp1$sigma))
      cat(sprintf("Comp B: p = %.2f, mu = %.2f, sigma = %.2f\n", res$comp2$p, res$comp2$mu, res$comp2$sigma))
  }
}

# Run a few diverse scenarios
n_samples <- 5e6 # Large enough to minimize sampling noise on M6

test_scenario("High Separation, Similar Variances", n_samples, 0.4, -5.0, 5.0, 1.0, 1.2)
test_scenario("Moderate Separation, Very Different Variances", n_samples, 0.25, 0.0, 3.0, 0.5, 4.0)
test_scenario("Nearly Identical Means, Extremely Different Variances", n_samples, 0.6, 2.0, 2.1, 0.1, 5.0)
test_scenario("Massive Scale (Unstandardized Stress Test)", n_samples, 0.7, 100.0, 250.0, 30.0, 50.0)
test_scenario("Tiny Scale (Microscopic values)", n_samples, 0.35, -0.01, 0.02, 0.005, 0.01)

