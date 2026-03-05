source("code_r/hardt_price_alg_C.R")
source("code_r/hardt_price_gmm_1d.R")
source("code_r/hardt_price_gmm_nd.R")

cat("\n=== Testing Bare Crash ===\n")
n <- 25000; d <- 4; p <- 0.4
mu1 <- c(-1, 0, 1, 0); mu2 <- c(1, 0, -1, 0)
sigma1 <- rep(1, d); sigma2 <- rep(1, d)
set.seed(42)
labels <- rbinom(n, 1, 1 - p)
X <- matrix(0, nrow = n, ncol = d)
for (j in 1:d) {
    if (length(which(labels==0)) > 0) X[which(labels==0), j] <- rnorm(length(which(labels==0)), mu1[j], sqrt(sigma1[j]))
    if (length(which(labels==1)) > 0) X[which(labels==1), j] <- rnorm(length(which(labels==1)), mu2[j], sqrt(sigma2[j]))
}

# Run raw
options(error=traceback)
res <- Reduce4DTo1D(X, epsilon = 0.6, delta = 0.05)
