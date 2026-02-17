# Benchmark Loop: ADMM vs CVXR (10 Iterations)
# Settings: n=200, p=3000, s=10
library(CVXR)
library(stats)
library(mclust) # For ARI

# Source both
source("code_r/sdp_kmeans.R")
source("code_r/sdp_kmeans_admm.R")

n_iter <- 10
results <- data.frame(
    Iter = integer(n_iter),
    CVXR_Time = numeric(n_iter),
    CVXR_ACC = numeric(n_iter),
    ADMM_Time = numeric(n_iter),
    ADMM_ACC = numeric(n_iter),
    ADMM_Objective = numeric(n_iter),
    CVXR_Objective = numeric(n_iter)
)

n <- 200
K <- 2
p <- 3000
s <- 10 # Sparsity
l2_sq_diff <- 16 # ||mu1 - mu2||^2
mu_val <- sqrt(4 / s)

cat(sprintf("Starting Benchmark Loop (10 Iterations, n=%d, p=%d)...\n", n, p))

for (i in 1:n_iter) {
    set.seed(42 + i) # Different seed per iteration

    cat(sprintf("\n--- Iteration %d/%d ---\n", i, n_iter))

    # Generate Data
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

    # 1. Run CVXR
    cat("Running CVXR...\n")
    t1 <- Sys.time()
    res_cvxr <- sdp_kmeans(G, K)
    t2 <- Sys.time()
    time_cvxr <- as.numeric(difftime(t2, t1, units = "secs"))

    # Acc CVXR
    acc_cvxr <- max(mean(res_cvxr$cluster == true_labels), mean(res_cvxr$cluster != true_labels))

    # 2. Run ADMM
    cat("Running ADMM...\n")
    t3 <- Sys.time()
    # Use defaults (optimized)
    res_admm <- sdp_kmeans_admm(G, K, max_iter = 1000, tol = 1e-4, verbose = FALSE)
    t4 <- Sys.time()
    time_admm <- as.numeric(difftime(t4, t3, units = "secs"))

    # Acc ADMM
    acc_admm <- max(mean(res_admm$cluster == true_labels), mean(res_admm$cluster != true_labels))

    # Store
    results$Iter[i] <- i
    results$CVXR_Time[i] <- time_cvxr
    results$CVXR_ACC[i] <- acc_cvxr
    results$CVXR_Objective[i] <- res_cvxr$value
    results$ADMM_Time[i] <- time_admm
    results$ADMM_ACC[i] <- acc_admm
    results$ADMM_Objective[i] <- res_admm$value

    cat(sprintf(
        "CVXR: %.2fs (ACC=%.4f) | ADMM: %.2fs (ACC=%.4f)\n",
        time_cvxr, acc_cvxr, time_admm, acc_admm
    ))
}

cat("\n=== Benchmark Results (Average over 10 Runs) ===\n")
cat(sprintf("CVXR Average Time: %.4f seconds\n", mean(results$CVXR_Time)))
cat(sprintf("ADMM Average Time: %.4f seconds\n", mean(results$ADMM_Time)))
cat(sprintf("Speedup: %.2fx\n", mean(results$CVXR_Time) / mean(results$ADMM_Time)))
cat(sprintf("CVXR Average ACC : %.4f\n", mean(results$CVXR_ACC)))
cat(sprintf("ADMM Average ACC : %.4f\n", mean(results$ADMM_ACC)))

print(results)
