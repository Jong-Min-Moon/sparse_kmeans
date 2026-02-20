library(Rcpp)
library(RSpectra)

# Load the compiled DLL
lib_path <- "code_r/proj_simplex.dll"
if (!file.exists(lib_path)) {
    stop("Pre-compiled library 'code_r/proj_simplex.dll' not found.")
}
if (!("proj_simplex" %in% names(getLoadedDLLs()))) dyn.load(lib_path)

# Source sdp_kmeans
# As sdp_kmeans.R uses dyn.load inside, we might need to be careful not to reload or conflict.
# It seems sdp_kmeans.R checks if loaded.
# Sourcing it to get the function.
source("code_r/sdp_kmeans.R")

set.seed(123)
n <- 200 # Using smaller n for quick benchmark, or larger?
K <- 3
p <- 50
cat(sprintf("Benchmarking sdp_kmeans with n=%d, K=%d\n", n, K))

# Generate data
X <- matrix(rnorm(n * p), n, p)
# Create Gram matrix
G <- X %*% t(X)

# Function to run benchmark
run_bench <- function(n_threads) {
    Sys.setenv(OMP_NUM_THREADS = n_threads)
    cat(sprintf("Running with %d thread(s)...\n", n_threads))

    start_time <- Sys.time()
    # Run for limited iterations to save time but enough to measure
    Res <- sdp_kmeans(G, K, max_iter = 50, verbose = FALSE)
    end_time <- Sys.time()

    as.numeric(end_time - start_time, units = "secs")
}

# 1 Thread
time1 <- run_bench(1)
cat(sprintf("Time (1 thread): %.4f s\n", time1))

# 4 Threads
time4 <- run_bench(4)
cat(sprintf("Time (4 threads): %.4f s\n", time4))

speedup <- time1 / time4
cat(sprintf("Speedup: %.2fx\n", speedup))
