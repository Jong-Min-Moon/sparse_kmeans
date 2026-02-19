# Comparison of SDP K-means and kernlab::specc (Baseline)
# Simulation settings from driver.R (p=400, n=500, rho=0.45)

library(MASS)
library(mclust)
library(kernlab)

# Source required functions
source("../../code_r/sdp_kmeans.R")

# Simulation Parameters
p <- 400
n <- 500
K <- 2
rho <- 0.45
n_reps <- 10

# Generate Covariance
Omega <- matrix(0, p, p)
for (i in 1:p) {
    Omega[i, i] <- 1
    if (i > 1) Omega[i, i - 1] <- rho
    if (i < p) Omega[i, i + 1] <- rho
}
Sigma <- solve(Omega)

# Signal Generation (Target: || Omega * DeltaMu ||^2 = 9)
S_0 <- 1:10
v <- rep(0, p)
v[S_0] <- 1
delta <- sqrt(9 / sum(v^2))
mu_diff <- as.numeric(Sigma %*% (v * delta))
mu1 <- mu_diff / 2
mu2 <- -mu_diff / 2
true_labels <- c(rep(1, n / 2), rep(2, n / 2))

acc_calc <- function(pred) {
    max(mean(pred == true_labels), mean(pred != true_labels))
}

results <- data.frame(rep = 1:n_reps, sdp_acc = 0, kernlab_acc = 0)

cat(sprintf("Running %d repetitions (kernlab vs SDP)...\n", n_reps))

for (i in 1:n_reps) {
    set.seed(42 + i)
    cat(sprintf("Rep %d... ", i))

    # Generate Data
    X1 <- mvrnorm(n / 2, mu1, Sigma)
    X2 <- mvrnorm(n / 2, mu2, Sigma)
    X <- t(rbind(X1, X2)) # p x n

    # 1. SDP K-means (Baseline assumes Identity Covariance)
    G <- crossprod(X)
    res_sdp <- sdp_kmeans(G, K, max_iter = 4000)
    results$sdp_acc[i] <- acc_calc(res_sdp$cluster)

    # 2. kernlab::specc
    # specc expects data with observations in rows (n x p)
    res_kernlab <- specc(t(X), centers = K)
    results$kernlab_acc[i] <- acc_calc(as.integer(res_kernlab))

    cat(sprintf("SDP: %.4f, kernlab: %.4f\n", results$sdp_acc[i], results$kernlab_acc[i]))
}

# Summary
cat("\n=== Baseline Comparison (10 reps) ===\n")
cat(sprintf("Mean SDP Accuracy:      %.4f (sd: %.4f)\n", mean(results$sdp_acc), sd(results$sdp_acc)))
cat(sprintf("Mean kernlab Accuracy: %.4f (sd: %.4f)\n", mean(results$kernlab_acc), sd(results$kernlab_acc)))

# Save result for plotting if needed
saveRDS(results, "baseline_kernlab_comparison_results.rds")
