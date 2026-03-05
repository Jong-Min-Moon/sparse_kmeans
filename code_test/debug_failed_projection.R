source("code_r/hardt_price_alg_C.R")
source("code_r/hardt_price_gmm_1d.R")
source("code_r/hardt_price_gmm_nd.R")

cat("\n=== Testing The Rejected Projection ===\n")
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


a <- c(-0.4472873, 0.6602731, 0.5320567, 0.1657081)
x_1d <- drop(X %*% a)
cat("Projection Variance: ", var(x_1d), "\n")
res_1d <- Recover1DMixture(x_1d, delta = 0.05)

cat("\nReturned Targets:\n")
cat("M1: ", res_1d$comp1$mu, "\n")
cat("M2: ", res_1d$comp2$mu, "\n")
