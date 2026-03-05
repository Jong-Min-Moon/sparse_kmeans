source("code_r/hardt_price_alg_C.R")
source("code_r/hardt_price_gmm_1d.R")
source("code_r/hardt_price_gmm_nd.R")

cat("\n=== Testing Candidate Node Centers ===\n")
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

mu_hat <- colMeans(X)
sigma_hat <- sqrt(estimate_d_dim_variance(X, 0.05))

cat("Mu Hat:", mu_hat, "\n")
cat("Sigma Hat:", sigma_hat, "\n")

c_const <- 0.6
epsilon <- 0.6

N_mu <- generate_N_mu(mu_hat, sigma_hat, epsilon, c_const)
cat("Generated Grid Elements:", nrow(N_mu), "\n")

# Check if mu1 is inside N_mu
best_dist <- Inf
closest <- NULL
for (i in 1:nrow(N_mu)) {
    dist <- max(abs(N_mu[i, ] - mu1))
    if (dist < best_dist) {
        best_dist <- dist
        closest <- N_mu[i, ]
    }
}

cat("L_inf distance to closest node for mu1:", best_dist, "\n")
cat("Closest Node:", closest, "\n")
cat("Expected Mu1:", mu1, "\n")
