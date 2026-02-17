# Benchmark for SDP K-Means (n=500)
library(CVXR)
library(stats)

# Source the function
source("code_r/sdp_kmeans.R")

set.seed(123)
n <- 500
K <- 2

cat(sprintf("Generating %dx%d Gram matrix...\n", n, n))
# Generate a random PSD matrix G
# G = X'X where X is p x n. Let p=50 just for generating G.
p <- 50
X <- matrix(rnorm(n * p), nrow = p, ncol = n)
G <- crossprod(X)

cat("Starting SDP K-Means benchmark...\n")
start_time <- Sys.time()
res <- sdp_kmeans(G, K)
end_time <- Sys.time()

duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat(sprintf("SDP K-Means (n=%d, K=%d) Runtime: %.2f seconds\n", n, K, duration))

if (is.null(res$cluster) || all(is.na(res$cluster))) {
    cat("Result: FAILED (NA values returned)\n")
} else {
    cat("Result: Success\n")
}
