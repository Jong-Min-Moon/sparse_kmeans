# Test script for proj_simplex with variable sum
library(Rcpp)

if (.Platform$OS.type == "windows") {
    dyn.load("code_r/proj_simplex_opt.dll")
} else {
    dyn.load("code_r/proj_simplex_opt.so")
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
print(rowSums(P1))

# Project with sum=5
P5 <- proj_simplex_rows_cpp(M, 5.0)
print("Projected (Sum=5):")
print(P5)
print(rowSums(P5))

if (all(abs(rowSums(P5) - 5.0) < 1e-6)) {
    cat("SUCCESS: Projection respects target sum 5.0\n")
} else {
    cat("FAILURE: Target sum mismatch\n")
}
