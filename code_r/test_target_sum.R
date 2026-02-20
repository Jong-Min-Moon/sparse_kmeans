# Test script for proj_simplex with variable sum
library(Rcpp)

if (.Platform$OS.type == "windows") {
    dyn.load("code_r/proj_simplex.dll")
} else {
    dyn.load("code_r/proj_simplex.so")
}

proj_simplex_rows_cpp <- function(Mat, target_sum = 1.0) {
    .Call("proj_simplex_rows_wrapper", Mat, target_sum)
}

# Create a test matrix
set.seed(123)
M <- matrix(rnorm(10), 2, 5)
print("Original Matrix:")
print(M)

# Project with sum=1
P1 <- proj_simplex_rows_cpp(M, 1.0)
print("Projected (Sum=1):")
print(P1)

# Check sums (Should be 1.0)
row_sums_1 <- rowSums(P1)
print(row_sums_1)

if (all(abs(row_sums_1 - 1.0) < 1e-6)) {
    cat("SUCCESS: Projection sums to 1.0\n")
} else {
    cat("FAILURE: Projection does NOT sum to 1.0\n")
}
