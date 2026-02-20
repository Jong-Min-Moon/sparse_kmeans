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
n_reps <- 2
mu_val <- sqrt(4 / s)

cat("Starting Benchmark: k_prime factor 10 vs 3\n")
cat(sprintf("Parameters: N=%d, P=%d, K=%d\n", n, p, K))

# Use 4 threads for this test, as we want to see if reducing k_prime helps on top of parallelization
Sys.setenv(OMP_NUM_THREADS = 4)

run_test <- function(k_factor) {
    cat(sprintf("\n--- Testing k_prime_factor = %d ---\n", k_factor))
    results <- data.frame()

    for (i in 1:n_reps) {
        set.seed(3000 + i)

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
        res <- sdp_kmeans(G, K, max_iter = 1000, tol = 1e-4, verbose = FALSE, k_prime_factor = k_factor)
        t_end <- Sys.time()
        runtime <- as.numeric(difftime(t_end, t_start, units = "secs"))

        acc <- 0
        if (!is.null(res$cluster)) {
            acc <- max(mean(res$cluster == true_labels), mean(res$cluster != true_labels))
        }

        results <- rbind(results, data.frame(
            factor = k_factor,
            rep = i,
            time = runtime,
            acc = acc,
            iter = res$iter
        ))
        cat(sprintf("  Rep %d: Time=%.2fs, Acc=%.2f, Iter=%d\n", i, runtime, acc, res$iter))
    }
    return(results)
}

res10 <- run_test(10)
res3 <- run_test(3)

all_res <- rbind(res10, res3)
agg <- aggregate(cbind(time, acc, iter) ~ factor, data = all_res, mean)
print(agg)

speedup <- agg$time[agg$factor == 10] / agg$time[agg$factor == 3]
cat(sprintf("\nSpeedup (10 -> 3): %.2fx\n", speedup))
