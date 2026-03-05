source("code_r/hardt_price_alg_C.R")
source("code_r/hardt_price_gmm_1d.R")
source("code_r/hardt_price_gmm_nd.R")

cat("\n=== Debugging 1D Baseline Coordinates ===\n")
# Generate 4-dimensional mixture data
n <- 25000
d <- 4
p <- 0.4
mu1 <- c(-1, 0, 1, 0)
mu2 <- c(1, 0, -1, 0)
sigma1 <- rep(1, d)
sigma2 <- rep(1, d)

set.seed(42)
labels <- rbinom(n, 1, 1 - p)
X <- matrix(0, nrow = n, ncol = d)
for (j in 1:d) {
    if (length(which(labels==0)) > 0) X[which(labels==0), j] <- rnorm(length(which(labels==0)), mu1[j], sqrt(sigma1[j]))
    if (length(which(labels==1)) > 0) X[which(labels==1), j] <- rnorm(length(which(labels==1)), mu2[j], sqrt(sigma2[j]))
}

cat("Input Column Means:", colMeans(X), "\n")

# Run 1D directly on dimension 2
cat("\nRunning Dimension 2 directly:\n")
x2 <- X[, 2]
res2 <- Recover1DMixture(x2, delta=0.05)
cat("Output:", res2$comp1$mu, ",", res2$comp2$mu, "\n")

cat("\nRunning Dimension 4 directly:\n")
x4 <- X[, 4]
res4 <- Recover1DMixture(x4, delta=0.05)
cat("Output:", res4$comp1$mu, ",", res4$comp2$mu, "\n")
