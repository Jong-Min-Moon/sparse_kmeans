library(stats)
library(mclust)

# Source ADMM solver
# Adjust path to source correctly from root or script location
if (file.exists("code_r/sdp_kmeans.R")) {
    source("code_r/sdp_kmeans.R")
} else {
    source("../../code_r/sdp_kmeans.R")
}

# Parameters
n <- 500
K <- 2
s <- 10
# Use a subset of p_values to save time, or just one large one to show impact
p_values <- c(3000, 5000)
n_reps <- 2 # Reduced reps for benchmark
mu_val <- sqrt(4 / s)

cat("Starting Benchmark Sim03: ADMM P-Scaling (1 vs Multiple Threads)\n")

# Function to run simulation
run_sim <- function(n_threads) {
    Sys.setenv(OMP_NUM_THREADS = n_threads)
    cat(sprintf("\n=== Running with %d thread(s) ===\n", n_threads))

    results <- data.frame()

    for (p in p_values) {
        cat(sprintf("  p = %d\n", p))

        for (i in 1:n_reps) {
            set.seed(2024 + i) # Consistent seed per rep

            # Generate Data (same as run_sim03.R)
            n1 <- n / 2
            n2 <- n / 2
            mu1 <- numeric(p)
            mu2 <- numeric(p)
            mu1[1:s] <- mu_val
            mu2[1:s] <- -mu_val

            # Transpose generation logic to match dimensions
            # The original script does:
            # X1 <- matrix(rnorm(n1 * p), nrow = p) + mu1
            # This constructs p x n1 matrix.
            # X <- cbind(X1, X2) is p x n
            # G <- crossprod(X) is n x n (t(X) %*% X)

            X1 <- matrix(rnorm(n1 * p), nrow = p) + mu1
            X2 <- matrix(rnorm(n2 * p), nrow = p) + mu2
            X <- cbind(X1, X2)
            G <- crossprod(X)
            true_labels <- c(rep(1, n1), rep(2, n2))

            t_start <- Sys.time()
            res <- sdp_kmeans(G, K, max_iter = 1000, tol = 1e-4, verbose = FALSE)
            t_end <- Sys.time()
            runtime <- as.numeric(difftime(t_end, t_start, units = "secs"))

            acc <- 0
            if (!is.null(res$cluster)) {
                acc <- max(mean(res$cluster == true_labels), mean(res$cluster != true_labels))
            }

            results <- rbind(results, data.frame(
                threads = n_threads,
                p = p,
                rep = i,
                time = runtime,
                acc = acc
            ))
            cat(sprintf("    Rep %d: %.2fs (Acc: %.2f)\n", i, runtime, acc))
        }
    }
    return(results)
}

# Run Benchmarks
res1 <- run_sim(1)
res4 <- run_sim(4)

# Combine and Summarize
all_res <- rbind(res1, res4)
agg_res <- aggregate(time ~ threads + p, data = all_res, mean)
print(agg_res)

# Calculate Speedup
wide_res <- reshape(agg_res, idvar = "p", timevar = "threads", direction = "wide")
wide_res$speedup <- wide_res$time.1 / wide_res$time.4
print(wide_res)
