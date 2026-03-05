source("code_r/hardt_price_alg_C.R")
source("code_r/hardt_price_gmm_1d.R")
source("code_r/hardt_price_gmm_nd.R")

cat("\n=== Tracing True Center Candidates ===\n")
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
    idx1 <- which(labels == 0)
    idx2 <- which(labels == 1)
    if (length(idx1) > 0) X[idx1, j] <- rnorm(length(idx1), mu1[j], sqrt(sigma1[j]))
    if (length(idx2) > 0) X[idx2, j] <- rnorm(length(idx2), mu2[j], sqrt(sigma2[j]))
}

# Emulate Step 1 + 2
sigma_hat <- sqrt(estimate_d_dim_variance(X, 0.05))
mu_hat <- numeric(d)
for (j in 1:d) { mu_hat[j] <- Recover1DMixture(X[, j], delta = 0.05)$comp1$mu }

cat("Theoretical Mu1:", mu1, "\n")
cat("Theoretical Mu2:", mu2, "\n")
cat("Coordinate Tracked Mu_hat:", mu_hat, "\n\n")

# Run 1 bounded projection manually
a <- sample_N9(1, d)[1, ]
cat("Random vector a:", a, "\n")

# True 1D projection expectations:
true_p1 <- sum(a * mu1)
true_p2 <- sum(a * mu2)
cat("True a^T mu1:", true_p1, "\n")
cat("True a^T mu2:", true_p2, "\n")

# Reconstructed 1D
x_1d <- drop(X %*% a)
res_1d <- Recover1DMixture(x_1d, delta = 0.05)
cat("1D Algr Recovered Targets:", res_1d$comp1$mu, "and", res_1d$comp2$mu, "\n\n")

# Check if true mu1 is legally inside the grid
step_size <- 0.6 * 0.5 * sigma_hat
dist_to_true1 <- max(abs(mu1 - mu_hat))
cat(sprintf("L_inf distance from mu_hat to True Mu1: %.4f (Grid Radius: %.4f)\n", dist_to_true1, 2*sigma_hat))

# Check thresholding
err1 <- abs(true_p1 - res_1d$comp1$mu)
err2 <- abs(true_p1 - res_1d$comp2$mu)
thresh <- 0.6 * 0.5 * sigma_hat / 2
cat(sprintf("Testing TRUE mu1 on threshold %.4f -> err1: %.4f, err2: %.4f\n", thresh, err1, err2))
if (err1 > thresh && err2 > thresh) cat("TRUE MU1 REJECTED!\n") else cat("TRUE MU1 ACCEPTED!\n")

