library(Rcpp)

# Load the compiled DLL
dyn.load("code_r/proj_simplex.dll")

# Wrapper function if not exported by Rcpp Attributes yet in the session
# But typically Rcpp exports are handled via RcppExports.R or similar in packages.
# Here we probably need to access the function via .Call since it's a raw DLL / source
# The C++ file has // [[Rcpp::export]], so sourceCpp would work, but we used R CMD SHLIB.
# So we need to define the R wrapper manually or use the one in the C++ file if we source it?
# The C++ file has a manual wrapper: proj_simplex_rows_wrapper

proj_simplex_rows <- function(Mat) {
    .Call("proj_simplex_rows_wrapper", Mat)
}

# Test Correctness
set.seed(123)
n <- 100
p <- 1000
Mat <- matrix(rnorm(n * p), n, p)

cat("Running parallel projection...\n")
start_time <- Sys.time()
Res <- proj_simplex_rows(Mat)
end_time <- Sys.time()
cat("Time taken:", end_time - start_time, "\n")

# Verify constraints
row_sums <- rowSums(Res)
min_val <- min(Res)

cat("Min row sum:", min(row_sums), "\n")
cat("Max row sum:", max(row_sums), "\n")
cat("Min value:", min_val, "\n")

if (abs(max(row_sums) - 1) < 1e-10 && abs(min(row_sums) - 1) < 1e-10 && min_val > -1e-10) {
    cat("Verification passed!\n")
} else {
    cat("Verification failed!\n")
}
