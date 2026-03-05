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
  cat(sprintf("\n======================================================\n"))
  cat(sprintf("--- Scenario: %s ---\n", name))
  cat(sprintf("True Parameters (n = %1.1e):\n", n))
  cat(sprintf("Comp 1: p = %.3f, mu = %.3f, sigma = %.3f\n", p, mu1, sigma1))
  cat(sprintf("Comp 2: p = %.3f, mu = %.3f, sigma = %.3f\n", 1-p, mu2, sigma2))
  
  data <- generate_gmm_data(n, p, mu1, mu2, sigma1, sigma2)
  
  res <- Recover1DMixture(data$x, delta = delta)
  
  cat(sprintf("\nRecovered Parameters:\n"))
  if (isTRUE(res$fallback)) {
      cat("Output: FALLBACK (Single Component)\n")
      cat(sprintf("Comp: p = %.3f, mu = %.3f, sigma = %.3f\n", res$comp1$p, res$comp1$mu, res$comp1$sigma))
  } else {
      cat(sprintf("Comp A: p = %.3f, mu = %.3f, sigma = %.3f\n", res$comp1$p, res$comp1$mu, res$comp1$sigma))
      cat(sprintf("Comp B: p = %.3f, mu = %.3f, sigma = %.3f\n", res$comp2$p, res$comp2$mu, res$comp2$sigma))
  }
}

n_samples <- 5e6 # 5 million to reduce sampling variance

test_scenario("High Class Imbalance", n_samples, 0.02, -3.0, 3.0, 1.0, 1.0)
test_scenario("Near Indistinguishability", n_samples, 0.4, -0.1, 0.1, 1.0, 1.1)
test_scenario("Complete Overlap (Fallback Trigger)", n_samples, 0.5, 0.0, 0.0, 1.0, 1.0)

# What about testing how convergence scales with sample size?
test_scenario("Testing O(1/eps^12) - n = 5 Million", 5e6, 0.3, -2.0, 2.0, 1.0, 1.0)
test_scenario("Testing O(1/eps^12) - n = 50 Million", 5e7, 0.3, -2.0, 2.0, 1.0, 1.0) # WARNING: memory heavy, but possible in R.
