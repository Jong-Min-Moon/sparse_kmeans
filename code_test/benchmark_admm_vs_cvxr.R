# Benchmark: ADMM vs CVXR
library(CVXR)
library(stats)
library(mclust) # For ARI

# Source both
source("code_r/sdp_kmeans.R")
source("code_r/sdp_kmeans_admm.R")

set.seed(123)
n <- 500
K <- 2
p <- 3000
s <- 10 # Sparsity
l2_sq_diff <- 16 # ||mu1 - mu2||^2

# Calculate mu per feature
# (mu - (-mu))^2 * s = 16
# (2mu)^2 * s = 16
# 4 * mu^2 * s = 16
# mu^2 = 4/s
mu_val <- sqrt(4 / s)

# Generate Data with Cluster Structure
cat(sprintf("Generating %dx%d Data with %d clusters (s=%d, Dist^2=%.1f)...\n", n, p, K, s, l2_sq_diff))
n1 <- n / 2
n2 <- n / 2

# Sparse Means
mu1 <- numeric(p)
mu2 <- numeric(p)
mu1[1:s] <- mu_val
mu2[1:s] <- -mu_val

X1 <- matrix(rnorm(n1 * p), nrow = p) + mu1
X2 <- matrix(rnorm(n2 * p), nrow = p) + mu2
X <- cbind(X1, X2)
G <- crossprod(X)
true_labels <- c(rep(1, n1), rep(2, n2))

# 1. Run ADMM (Fast)
cat("\n--- Running ADMM Solver (n=500) ---\n")
start_admm <- Sys.time()
res_admm <- sdp_kmeans_admm(G, K, max_iter = 200, tol = 1e-4, verbose = TRUE)
end_admm <- Sys.time()
time_admm <- as.numeric(difftime(end_admm, start_admm, units = "secs"))

ari_admm <- adjustedRandIndex(res_admm$cluster, true_labels)
cat(sprintf("ADMM Runtime: %.2f seconds\n", time_admm))
cat(sprintf("ADMM Objective: %.2f\n", res_admm$value))

# Calculate Clustering Accuracy (ACC) for K=2
# Check alignment: 1->1, 2->2 vs 1->2, 2->1
acc_1 <- mean(res_admm$cluster == true_labels)
acc_2 <- mean(res_admm$cluster != true_labels) # If classes are flipped (for K=2 only)
acc_admm <- max(acc_1, acc_2)

ari_admm <- adjustedRandIndex(res_admm$cluster, true_labels)
cat(sprintf("ADMM Accuracy (ACC): %.4f\n", acc_admm))
cat(sprintf("ADMM ARI vs Truth: %.4f\n", ari_admm))

# 2. Run CVXR (Slow - Optional)
cat("\n--- Accuracy Check (n=200) ---\n")
n_small <- 200
n_small_half <- n_small / 2
X_small <- X[, c(1:n_small_half, (n1 + 1):(n1 + n_small_half))]
G_small <- crossprod(X_small)
true_labels_small <- c(rep(1, n_small_half), rep(2, n_small_half))

cat("Running CVXR (n=200)...\n")
t1 <- Sys.time()
res_cvxr <- sdp_kmeans(G_small, K)
t2 <- Sys.time()
ari_cvxr <- adjustedRandIndex(res_cvxr$cluster, true_labels_small)
cat(sprintf(
    "CVXR (n=200) Time: %.2f s | Obj: %.2f | ARI: %.4f\n",
    as.numeric(difftime(t2, t1, units = "secs")), res_cvxr$value, ari_cvxr
))

cat("Running ADMM (n=200)...\n")
t3 <- Sys.time()
res_admm_small <- sdp_kmeans_admm(G_small, K, max_iter = 500, tol = 1e-5)
t4 <- Sys.time()
ari_admm_small <- adjustedRandIndex(res_admm_small$cluster, true_labels_small)
cat(sprintf(
    "ADMM (n=200) Time: %.2f s | Obj: %.2f | ARI: %.4f\n",
    as.numeric(difftime(t4, t3, units = "secs")), res_admm_small$value, ari_admm_small
))

# Calculate Accuracy for Verification
acc_small_1 <- mean(res_admm_small$cluster == true_labels_small)
acc_small_2 <- mean(res_admm_small$cluster != true_labels_small)
acc_admm_small <- max(acc_small_1, acc_small_2)
cat(sprintf("ADMM (n=200) Accuracy (ACC): %.4f\n", acc_admm_small))

# Compare Objectives & ARI
diff_obj <- abs(res_cvxr$value - res_admm_small$value)
ari_diff <- abs(ari_cvxr - ari_admm_small)
cat(sprintf("Objective Difference: %.4f\n", diff_obj))
cat(sprintf("ARI Difference: %.4f\n", ari_diff))

if (ari_admm_small > 0.95 && ari_diff < 0.05) {
    cat("ACCURACY CHECK PASSED: ADMM provides high accuracy.\n")
} else {
    cat("ACCURACY CHECK WARNING: ADMM accuracy might differ.\n")
}
