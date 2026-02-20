# Test ISEE Noisy (Small Scale)

source("ISEE_kmeans_noisy.R")

# Generate small data
set.seed(123)
n <- 20
p <- 10
K <- 2

# Simple mean separation on first 2 features
x <- matrix(rnorm(n * p), nrow = p, ncol = n)
x[1:2, 1:(n/2)] <- x[1:2, 1:(n/2)] + 2
x[1:2, (n/2 + 1):n] <- x[1:2, (n/2 + 1):n] - 2

# Run ISEE K-Means Noisy
cat("Running test...\n")
tryCatch({
  res <- ISEE_kmeans_noisy(x, K, n_iter = 3, stable_iter = 2)
  print(res$cluster)
  print(res$iter)
  cat("Test Passed.\n")
}, error = function(e) {
  cat("Test Failed: ", e$message, "\n")
})
