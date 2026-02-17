library(stats)
library(mclust)

# Source ADMM solver
if (file.exists("code_r/sdp_kmeans.R")) {
    source("code_r/sdp_kmeans.R")
} else {
    source("../../code_r/sdp_kmeans.R")
}

# Parameters
n <- 200
K <- 2
s <- 10
p <- 3000 # Focusing on P=3000 as per recent context
n_reps <- 100
mu_val <- sqrt(4 / s)

cat("Starting Simulation 03: 100 Reps (Tol=1e-3, MaxIter=1000, Sum=K)\n")
cat(sprintf("N=%d, P=%d, K=%d, s=%d, Reps=%d\n", n, p, K, s, n_reps))

# Use parallel threads for the solver itself
Sys.setenv(OMP_NUM_THREADS = 4)

results <- data.frame(
    rep = integer(),
    time = numeric(),
    acc = numeric(),
    ari = numeric(),
    obj = numeric(),
    iter = integer()
)

set.seed(2024) # Re-using the "good" seed sequence start

for (i in 1:n_reps) {
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

    t_start <- Sys.time()
    # k_prime_factor defaults to 3 now in sdp_kmeans.R, but being explicit is good.
    res <- sdp_kmeans(G, K, max_iter = 1000, tol = 1e-3, verbose = TRUE, k_prime_factor = 3, report_interval = 500)
    dur <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))

    acc <- 0
    if (!is.null(res$cluster)) {
        acc <- max(mean(res$cluster == true_labels), mean(res$cluster != true_labels))
    }
    ari <- mclust::adjustedRandIndex(res$cluster, true_labels)

    results <- rbind(results, data.frame(
        rep = i,
        time = dur,
        acc = acc,
        ari = ari,
        obj = res$value,
        iter = res$iter
    ))

    cat(sprintf("Rep %d/%d | Time: %.2fs | Acc: %.4f\n", i, n_reps, dur, acc))
}

cat("\n=== Final Summary (100 Reps) ===\n")
mean_acc <- mean(results$acc)
sd_acc <- sd(results$acc)
mean_time <- mean(results$time)
mean_iter <- mean(results$iter)

cat(sprintf("Mean Accuracy: %.4f (SD: %.4f)\n", mean_acc, sd_acc))
cat(sprintf("Mean Time: %.2fs\n", mean_time))
cat(sprintf("Mean Iterations: %.1f\n", mean_iter))

write.csv(results, "simulations/sim03_admm_p_scaling/results_sim03_100reps_tol_1e3.csv", row.names = FALSE)
cat("Detailed results saved to simulations/sim03_admm_p_scaling/results_sim03_100reps_tol_1e3.csv\n")
