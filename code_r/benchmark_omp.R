library(Rcpp)

# Load the compiled DLL
if (file.exists("code_r/proj_simplex.dll")) {
    dyn.load("code_r/proj_simplex.dll")
} else {
    dyn.load("proj_simplex.dll")
}

proj_simplex_rows <- function(Mat) {
    .Call("proj_simplex_rows_wrapper", Mat)
}

set.seed(123)
# Use a substantial size to see the difference
n <- 5000
p <- 1000
cat(sprintf("Benchmarking with matrix size: %d x %d\n", n, p))
Mat <- matrix(rnorm(n * p), n, p)

# Function to run benchmark
run_bench <- function(n_threads) {
    Sys.setenv(OMP_NUM_THREADS = n_threads)
    cat(sprintf("Running with %d thread(s)...\n", n_threads))

    # Warmup
    garbage <- proj_simplex_rows(Mat[1:10, ])

    start_time <- Sys.time()
    Res <- proj_simplex_rows(Mat)
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
