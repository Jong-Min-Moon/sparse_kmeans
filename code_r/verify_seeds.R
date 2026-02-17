library(stats)
library(mclust)

# Source ADMM solver
if (file.exists("code_r/sdp_kmeans.R")) {
    source("code_r/sdp_kmeans.R")
} else {
    source("../../code_r/sdp_kmeans.R")
}

# Parameters matching sim03
n <- 200
p <- 3000
K <- 2
s <- 10
mu_val <- sqrt(4 / s)

cat("Verifying Seeds from Sim03 (Expected: Acc ~0.64, 0.675)\n")

# Logic from run_sim03.R
set.seed(2024)

# p loop starts with 3000
# We want to replicate the first few reps
n_reps_to_test <- 3

for (i in 1:n_reps_to_test) {
    # Data Gen (Exactly as in run_sim03.R)
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

    cat(sprintf("\n--- Rep %d ---\n", i))

    # Test with default k_prime_factor = 3
    cat("  Running with k_prime_factor = 3 (New Default)...\n")
    t1 <- Sys.time()
    res3 <- sdp_kmeans(G, K, max_iter = 1000, tol = 1e-4, verbose = FALSE, k_prime_factor = 3)
    dur3 <- as.numeric(Sys.time() - t1)
    acc3 <- max(mean(res3$cluster == true_labels), mean(res3$cluster != true_labels))
    cat(sprintf("    Acc: %.4f | Time: %.2fs | Iter: %d\n", acc3, dur3, res3$iter))

    # Test with k_prime_factor = 10 (Old Behavior)
    cat("  Running with k_prime_factor = 10 (Old Behavior)...\n")
    t2 <- Sys.time()
    res10 <- sdp_kmeans(G, K, max_iter = 1000, tol = 1e-4, verbose = FALSE, k_prime_factor = 10)
    dur10 <- as.numeric(Sys.time() - t2)
    acc10 <- max(mean(res10$cluster == true_labels), mean(res10$cluster != true_labels))
    cat(sprintf("    Acc: %.4f | Time: %.2fs | Iter: %d\n", acc10, dur10, res10$iter))
}
