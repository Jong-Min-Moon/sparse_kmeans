library(MASS)
source("code_r/hardt_price_gmm_1d.R")
source("code_r/hardt_price_gmm_nd.R")
source("code_r/hardt_price_alg_B.R")

cat("=======================================================\n")
cat("=== Benchmarking Alg B (d-to-4) on d=6 Mixture ===\n")
cat("=======================================================\n")

set.seed(42)
d <- 6
n <- 5000

# Ground Truth Parameters (Simple geometry for speed profiling)
mu1 <- rep(-2, d)
mu2 <- rep(2, d)

S1 <- diag(1, d)
S2 <- diag(2, d)

# Generate Data
cat("[BENCHMARK] Generating 6D Input Data...\n")
X1 <- mvrnorm(n * 0.5, mu1, S1)
X2 <- mvrnorm(n * 0.5, mu2, S2)
X <- rbind(X1, X2)

cat(sprintf("[BENCHMARK] Input Data Dimensions: %d samples, %d features\n", nrow(X), ncol(X)))

# Run Recovery!
cat("\n[BENCHMARK] Executing ReduceDTo4... (Tracking Processing Time)\n")

start_time <- Sys.time()

# We use a relatively large epsilon/delta for benching the algorithm framework overhead
# instead of blowing out infinite grids for precision
res <- ReduceDTo4(X, epsilon = 0.5, delta = 0.1)

end_time <- Sys.time()
processing_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat(sprintf("\n[BENCHMARK] Summary: Recovery Completed in %.2f seconds.\n", processing_time))

# Just check the first few coordinates for sanity
cat(sprintf("Sampled rec_mu A: %.3f %.3f %.3f\n", res$comp1$mu[1], res$comp1$mu[2], res$comp1$mu[3]))
cat(sprintf("Sampled rec_mu B: %.3f %.3f %.3f\n", res$comp2$mu[1], res$comp2$mu[2], res$comp2$mu[3]))
