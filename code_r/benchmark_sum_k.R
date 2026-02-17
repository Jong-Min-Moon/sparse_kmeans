library(stats)
library(mclust)

# Source ADMM solver (which now uses K)
if (file.exists("code_r/sdp_kmeans.R")) {
    source("code_r/sdp_kmeans.R")
} else {
    source("../../code_r/sdp_kmeans.R")
}

# Parameters
n <- 200
K <- 2
s <- 10
p <- 3000
mu_val <- sqrt(4 / s)

cat("Benchmarking with Simplex Sum = K (K=2)\n")

# Same seed as successful verification
set.seed(2024)

# Data Gen
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

t1 <- Sys.time()
res <- sdp_kmeans(G, K, max_iter = 1000, tol = 1e-4, verbose = FALSE, k_prime_factor = 3)
dur <- as.numeric(Sys.time() - t1)

acc <- 0
if (!is.null(res$cluster)) {
    acc <- max(mean(res$cluster == true_labels), mean(res$cluster != true_labels))
}

cat(sprintf("Acc: %.4f | Time: %.2fs | Iter: %d\n", acc, dur, res$iter))
cat("Checking row sums of Y (should be ~K=2)...\n")
# We need to access Y from the result if we returned it, wait sdp_kmeans doesn't return Y.
# It returns Z. Z matches Y at convergence.
# Let's check row sums of Z.
row_sums_Z <- rowSums(res$Z)
cat(sprintf("Mean Row Sum of Z: %.4f (Target: %.4f)\n", mean(row_sums_Z), as.numeric(K)))
print(summary(row_sums_Z))
