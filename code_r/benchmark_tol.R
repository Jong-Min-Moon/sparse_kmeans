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
p <- 3000
n_reps <- 5 # Increased reps for better average
mu_val <- sqrt(4 / s)

cat("Starting Benchmark: Tolerance 1e-2 vs 1e-4\n")
cat(sprintf("Parameters: N=%d, P=%d, K=%d, k_prime_factor=3 (default)\n", n, p, K))

Sys.setenv(OMP_NUM_THREADS = 4)

run_test <- function(tol_val) {
    cat(sprintf("\n--- Testing tol = %.0e ---\n", tol_val))
    results <- data.frame()

    for (i in 1:n_reps) {
        set.seed(4000 + i)

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

        t_start <- Sys.time()
        # Using default k_prime_factor = 3 (from updated file)
        # Explicitly using max_iter = 10000 to match new default intent
        res <- sdp_kmeans(G, K, max_iter = 10000, tol = tol_val, verbose = FALSE)
        t_end <- Sys.time()
        runtime <- as.numeric(difftime(t_end, t_start, units = "secs"))

        acc <- 0
        if (!is.null(res$cluster)) {
            acc <- max(mean(res$cluster == true_labels), mean(res$cluster != true_labels))
        }

        results <- rbind(results, data.frame(
            tol = tol_val,
            rep = i,
            time = runtime,
            acc = acc,
            iter = res$iter
        ))
        cat(sprintf("  Rep %d: Time=%.2fs, Acc=%.2f, Iter=%d\n", i, runtime, acc, res$iter))
    }
    return(results)
}

res_loose <- run_test(1e-2)
res_tight <- run_test(1e-4)

all_res <- rbind(res_loose, res_tight)
agg <- aggregate(cbind(time, acc, iter) ~ tol, data = all_res, mean)
print(agg)

speedup <- agg$time[agg$tol == 1e-4] / agg$time[agg$tol == 1e-2]
cat(sprintf("\nSpeedup (1e-4 -> 1e-2): %.2fx\n", speedup))
