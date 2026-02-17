# Simulation 03: ADMM Performance Scaling with p
# Settings: N=200, p in {3000, 4000, 5000}, s=10, 30 Reps
# Output: Accuracy and Runtime

library(stats)
library(mclust)

# Source ADMM solver
source("code_r/sdp_kmeans_admm.R")

# Parameters
n <- 200
K <- 2
s <- 10
p_values <- c(3000, 4000, 5000)
n_reps <- 30
mu_val <- sqrt(4 / s) # Signal strength

# Results container
results <- data.frame(
    p = integer(),
    rep = integer(),
    time = numeric(),
    acc = numeric(),
    ari = numeric(),
    objective = numeric()
)

cat("Starting Simulation 03: ADMM P-Scaling\n")
cat(sprintf("N=%d, s=%d, Reps=%d\n", n, s, n_reps))

set.seed(2024)

for (p in p_values) {
    cat(sprintf("\n--- Running for p = %d ---\n", p))

    for (i in 1:n_reps) {
        # Generate Data
        n1 <- n / 2
        n2 <- n / 2
        mu1 <- numeric(p)
        mu2 <- numeric(p)
        mu1[1:s] <- mu_val
        mu2[1:s] <- -mu_val

        # Add noise
        X1 <- matrix(rnorm(n1 * p), nrow = p) + mu1
        X2 <- matrix(rnorm(n2 * p), nrow = p) + mu2
        X <- cbind(X1, X2)
        G <- crossprod(X)
        true_labels <- c(rep(1, n1), rep(2, n2))

        # Run ADMM
        t_start <- Sys.time()
        res <- sdp_kmeans_admm(G, K, max_iter = 1000, tol = 1e-4, verbose = FALSE)
        t_end <- Sys.time()
        runtime <- as.numeric(difftime(t_end, t_start, units = "secs"))

        # Calculate Metrics
        acc <- max(mean(res$cluster == true_labels), mean(res$cluster != true_labels))
        ari <- mclust::adjustedRandIndex(res$cluster, true_labels)

        # Store
        results <- rbind(results, data.frame(
            p = p,
            rep = i,
            time = runtime,
            acc = acc,
            ari = ari,
            objective = res$value
        ))

        cat(sprintf("p=%d | Rep %d/%d | Time: %.2fs | ACC: %.4f\n", p, i, n_reps, runtime, acc))
    }
}

# Save Results
out_file <- "simulations/sim03_admm_p_scaling/results_sim03.csv"
write.csv(results, out_file, row.names = FALSE)
cat(sprintf("\nSimulation Complete. Results saved to %s\n", out_file))

# Summary
cat("\n=== Summary Stats ===\n")
agg_res <- aggregate(cbind(time, acc, ari) ~ p, data = results, mean)
print(agg_res)
