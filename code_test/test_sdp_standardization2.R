library(MASS)
library(mclust)
source("code_r/sdp_kmeans.R")
source("code_r/get_cluster_acc.R")

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
  
  # K-MEANS BASELINE
  km_unstd <- kmeans(X, K, nstart=20)
  acc_km_unstd <- get_cluster_acc(labels, km_unstd$cluster)
  
  km_std <- kmeans(scale(X), K, nstart=20)
  acc_km_std <- get_cluster_acc(labels, km_std$cluster)
  
  cat(sprintf("[K-Means] Accuracy (Unstandardized): %.4f\n", acc_km_unstd))
  cat(sprintf("[K-Means] Accuracy (Standardized):   %.4f\n", acc_km_std))

  # SPECTRAL CLUSTERING BASELINE
  G_unstd <- tcrossprod(X)
  eig_unstd <- RSpectra::eigs_sym(G_unstd, K, which = "LA")
  sc_unstd <- kmeans(eig_unstd$vectors, centers = K, nstart = 20)
  acc_sc_unstd <- get_cluster_acc(labels, sc_unstd$cluster)

  X_std <- scale(X)
  G_std <- tcrossprod(X_std)
  eig_std <- RSpectra::eigs_sym(G_std, K, which = "LA")
  sc_std <- kmeans(eig_std$vectors, centers = K, nstart = 20)
  acc_sc_std <- get_cluster_acc(labels, sc_std$cluster)

  cat(sprintf("[Spectral] Accuracy (Unstandardized): %.4f\n", acc_sc_unstd))
  cat(sprintf("[Spectral] Accuracy (Standardized):   %.4f\n", acc_sc_std))

  # Evaluate Unstandardized
  res_unstd <- sdp_kmeans(G_unstd, K, verbose=FALSE, max_iter=5000)
  acc_unstd <- get_cluster_acc(labels, res_unstd$cluster)
  
  # Evaluate Standardized
  res_std <- sdp_kmeans(G_std, K, verbose=FALSE, max_iter=5000)
  acc_std <- get_cluster_acc(labels, res_std$cluster)
  
  cat(sprintf("[SDP K-Means] Accuracy (Unstandardized): %.4f\n", acc_unstd))
  cat(sprintf("[SDP K-Means] Accuracy (Standardized):   %.4f\n", acc_std))
}

# Scenario A: Signal is in the low-variance feature
Sigma_A <- diag(c(20, 1, rep(1, p-2)))
mu1_A <- c(0, 3, rep(0, p-2))
mu2_A <- c(0, -3, rep(0, p-2))
run_simulation(mu1_A, mu2_A, Sigma_A, "Signal in low-variance feature (Standardization should help)")

# Scenario B: Signal is in the high-variance feature
Sigma_B <- diag(c(20, 1, rep(1, p-2)))
mu1_B <- c(10, 0, rep(0, p-2))
mu2_B <- c(-10, 0, rep(0, p-2))
run_simulation(mu1_B, mu2_B, Sigma_B, "Signal in high-variance feature (Standardization should harm)")
