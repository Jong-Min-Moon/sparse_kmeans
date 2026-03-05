source("code_r/hardt_price_gmm_1d.R")

cat("\n=== Testing Master Algorithm 3.3 Orchestrator ===\n")
set.seed(42)
n <- 5000000

test_scenario <- function(name, p1_true, p2_true, mu1_true, mu2_true, sigma1_true, sigma2_true) {
  cat(sprintf("\n--- Scenario: %s ---\n", name))
  
  labels <- sample(1:2, n, replace = TRUE, prob = c(p1_true, p2_true))
  x <- numeric(n)
  n1 <- sum(labels == 1)
  n2 <- n - n1
  x[labels == 1] <- rnorm(n1, mean = mu1_true, sd = sigma1_true)
  x[labels == 2] <- rnorm(n2, mean = mu2_true, sd = sigma2_true)
  
  cat("True Parameters:\n")
  cat(sprintf("Comp 1: p = %.2f, mu = %.2f, sigma = %.2f\n", p1_true, mu1_true, sigma1_true))
  cat(sprintf("Comp 2: p = %.2f, mu = %.2f, sigma = %.2f\n", p2_true, mu2_true, sigma2_true))
  
  # Recover using the Master Orchestrator (Algorithm 3.3)
  res <- Recover1DMixture(x, delta = 0.05)
  
  cat("Recovered Parameters:\n")
  if (!is.null(res$fallback)) {
      cat("Algorithm 3.3 Route: Output a single merged Gaussian cluster (Means and Variances inseparable)\n")
  }
  
  # Automatically sort components by mean for easier visual comparison to true
  if (res$comp1$mu > res$comp2$mu) {
      c_temp <- res$comp1; res$comp1 <- res$comp2; res$comp2 <- c_temp
  }
  
  cat(sprintf("Comp A: p = %.2f, mu = %.2f, sigma = %.2f\n", res$comp1$p, res$comp1$mu, res$comp1$sigma))
  cat(sprintf("Comp B: p = %.2f, mu = %.2f, sigma = %.2f\n", res$comp2$p, res$comp2$mu, res$comp2$sigma))
}

# 1. Separated Means 
test_scenario("Well-Separated Means (Algorithm 3.1 Route)", 0.3, 0.7, -2.0, 5.0, 0.5, 1.5)

# 2. Identical Means, Different Variances
test_scenario("Identical Means, Different Variances (Algorithm 3.2 Route)", 0.6, 0.4, 3.5, 3.5, 1.0, 4.0)

# 3. Completely Inseparable (Same Means, Same Variances)
test_scenario("Inseparable Components (Fallback Route)", 0.5, 0.5, 0.0, 0.0, 1.0, 1.0)
