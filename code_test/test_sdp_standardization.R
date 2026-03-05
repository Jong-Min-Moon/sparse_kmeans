library(MASS)
library(mclust)
source("code_r/sdp_kmeans.R")

set.seed(42)
n <- 200
p <- 10
K <- 2

run_simulation <- function(mu1, mu2, Sigma, scenario_name) {
  cat("\n======================================================\n")
  cat("Scenario:", scenario_name, "\n")
  cat("======================================================\n")
  
  labels <- c(rep(1, n/2), rep(2, n/2))
  X <- matrix(0, n, p)
  for (i in 1:n) {
    if (labels[i] == 1) {
      X[i, ] <- mvrnorm(1, mu1, Sigma)
    } else {
      X[i, ] <- mvrnorm(1, mu2, Sigma)
    }
  }
  
  # Evaluate Unstandardized
  G_unstd <- tcrossprod(X)
  res_unstd <- sdp_kmeans(G_unstd, K, verbose=FALSE)
  ari_unstd <- adjustedRandIndex(labels, res_unstd$cluster)
  
  # Evaluate Standardized
  X_std <- scale(X)
  G_std <- tcrossprod(X_std)
  res_std <- sdp_kmeans(G_std, K, verbose=FALSE)
  ari_std <- adjustedRandIndex(labels, res_std$cluster)
  
  cat(sprintf("ARI (Unstandardized): %.4f\n", ari_unstd))
  cat(sprintf("ARI (Standardized):   %.4f\n", ari_std))
  
  return(list(ari_unstd=ari_unstd, ari_std=ari_std))
}

# ------------------------------------------------------------------------------
# Scenario A: Signal is in the low-variance feature
# Standardization should HELP here.
# ------------------------------------------------------------------------------
Sigma_A <- diag(c(20, 1, rep(1, p-2)))
mu1_A <- c(0, 3, rep(0, p-2))
mu2_A <- c(0, -3, rep(0, p-2))
run_simulation(mu1_A, mu2_A, Sigma_A, "Signal in low-variance feature")

# ------------------------------------------------------------------------------
# Scenario B: Signal is in the high-variance feature
# Standardization might HARM here.
# ------------------------------------------------------------------------------
Sigma_B <- diag(c(20, 1, rep(1, p-2)))
mu1_B <- c(10, 0, rep(0, p-2))
mu2_B <- c(-10, 0, rep(0, p-2))
run_simulation(mu1_B, mu2_B, Sigma_B, "Signal in high-variance feature")
