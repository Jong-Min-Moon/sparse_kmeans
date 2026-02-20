library(stats)
library(mclust)

# Source ADMM solver
if (file.exists("code_r/sdp_kmeans.R")) {
    source("code_r/sdp_kmeans.R")
} else {
    source("../../code_r/sdp_kmeans.R")
}

# Parameters matching poor performance case
n <- 200
K <- 2
s <- 10
p <- 3000
mu_val <- sqrt(4 / s)

cat("Investigating Poor Clustering Performance\n")

# Generate Data (Consistent Seed)
set.seed(4001) # Use a seed that might have failed
n1 <- n / 2
n2 <- n / 2
mu1 <- numeric(p)
mu2 <- numeric(p)
mu1[1:s] <- mu_val
mu2[1:s] <- -mu_val

X1 <- matrix(rnorm(n1 * p), nrow = p) + mu1
X2 <- matrix(rnorm(n2 * p), nrow = p) + mu2
X <- cbind(X1, X2)
G <- crossprod(X)
true_labels <- c(rep(1, n1), rep(2, n2))

cat("Running sdp_kmeans with verbose=TRUE...\n")
# Run with VERBOSE to see residual progress
res <- sdp_kmeans(G, K, max_iter = 1000, tol = 1e-4, verbose = TRUE, k_prime_factor = 3)

acc <- max(mean(res$cluster == true_labels), mean(res$cluster != true_labels))
cat(sprintf("Final Accuracy: %.2f\n", acc))
cat(sprintf("Total Iterations: %d\n", res$iter))
