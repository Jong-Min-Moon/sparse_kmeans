# Test Block Coordinate Optimization with Thompson Sampling (Known Covariance)

source("cluster_thompson.R")

# Generate Synthetic Data (Small scale for quick test)
set.seed(123)
n <- 40
p <- 20
K <- 2

# Cluster centers on first 5 features
mu1 <- rep(0, p)
mu2 <- rep(0, p)
mu2[1:5] <- 3 # Strong signal

X1 <- matrix(rnorm(n/2 * p, mean=0), nrow=n/2)
X2 <- matrix(rnorm(n/2 * p, mean=0) + matrix(rep(mu2, n/2), nrow=n/2, byrow=TRUE), nrow=n/2)
X <- rbind(X1, X2)
X <- t(X) # p x n

true_labels <- c(rep(1, n/2), rep(2, n/2))

cat("Data Summary:\n")
cat(sprintf("n=%d, p=%d, Signal on features 1-5\n", n, p))

# Run Bandit Algorithm (Known Covariance version)
cat("\nRunning block_coordinate_optim_thompson...\n")
# Assuming Diagonal Covariance for testing solve(cov, X)
diag_cov <- diag(p)
res <- cluster_thompson(X, K, n_iter = 100, C = 0.5, FDR_level = 0.4, n_perms = 50, covariance = diag_cov)

# Evaluate
cat("\n--- Results ---\n")
print(res$selected)
cat(sprintf("Selected Features: %s\n", paste(which(res$selected), collapse=", ")))

# Check accuracy
match_acc <- max(mean(res$cluster == true_labels), mean(res$cluster != true_labels)) 
cat(sprintf("Clustering Accuracy (Approx): %.2f%%\n", match_acc * 100))

# Check Alpha/Beta
cat("\n--- Posterior Params (First 10) ---\n")
print(rbind(Alpha=res$alpha[1:10], Beta=res$beta[1:10]))
