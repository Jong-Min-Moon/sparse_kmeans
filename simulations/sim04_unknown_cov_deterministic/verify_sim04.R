library(MASS)
library(mclust)
library(Matrix)
library(foreach)
library(doParallel)
library(glmnet)
library(kernlab)

# Source dependencies
source("../../code_r/block_coordinate_optim_deterministic_unknowncov.R")
source("../../code_r/ISEE_bicluster.R")
source("../../code_r/get_intercept_residual_lasso.R")
source("../../code_r/clustering_block_knowncov.R")
source("../../code_r/selection_block_greedy_screening.R")
source("../../code_r/sdp_kmeans.R")
source("../../code_r/utils.R")
source("../../code_r/cluster_spectral_matlab.R")

# Register Parallel
if (!exists("cl_verified")) {
    cl_verified <- makeCluster(max(1, parallel::detectCores() - 1))
    registerDoParallel(cl_verified)
}

# Parameters as requested: p=100, rho=0.2
set.seed(42)
p <- 100
n <- 200
S_0 <- 1:10
rho <- 0.2
separation <- 4

cat(sprintf("Generating data (MATLAB logic): p=%d, n=%d, rho=%.2f, sep=%.1f\n", p, n, rho, separation))

# 1. Precision Matrix Omega (Tridiagonal)
Omega <- matrix(0, p, p)
for (i in 1:p) {
    Omega[i, i] <- 1
    if (i > 1) Omega[i, i - 1] <- rho
    if (i < p) Omega[i, i + 1] <- rho
}
Sigma <- solve(Omega)

# 2. MATLAB Magnitude Logic
# magnitude = separation / 2 / sqrt( sum( Sigma[support, support] ) )
Sigma_S0 <- Sigma[S_0, S_0]
sum_Sigma_S0 <- sum(Sigma_S0) # MATLAB sum(..., "all")
magnitude <- separation / 2 / sqrt(sum_Sigma_S0)

# 3. Symmetric Means
# s_vec = indicator on support
s_vec <- rep(0, p)
s_vec[S_0] <- 1
mu_0 <- Sigma %*% (magnitude * s_vec)
mu1 <- as.numeric(-mu_0)
mu2 <- as.numeric(mu_0)

cat(sprintf("Calculated magnitude: %.4f, Signal Strength (Strength): %.4f\n", magnitude, 40 * magnitude^2))

# 4. Generate Data
X1 <- MASS::mvrnorm(n / 2, mu1, Sigma)
X2 <- MASS::mvrnorm(n / 2, mu2, Sigma)
X <- t(rbind(X1, X2))
true_labels <- c(rep(1, n / 2), rep(2, n / 2))

cat("Running Deterministic Algorithm...\n")
res <- block_coordinate_optim_deterministic_unknowncov(X, K = 2, n_iter = 20, stable_iter = 5, true_labels = true_labels)

cat(sprintf("ARI: %.4f\n", mclust::adjustedRandIndex(res$cluster, true_labels)))
cat(sprintf("Features: %d selected. Signal captured: %d/10\n", sum(res$selected_features), sum(res$selected_features[1:10])))
