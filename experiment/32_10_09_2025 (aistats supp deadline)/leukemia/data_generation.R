generate_isotropic_Gaussian <- function(sep, s, p, n, seed) {
  K <- 2
  cluster_true <- c(rep(1, n/2), rep(2, n/2))
  M <- 0.5 * sep / sqrt(s)
  sparse_mean <- c(rep(1, s), rep(0, p - s))
  mu_1 <- -M * sparse_mean
  mu_2 <-  M * sparse_mean
  x_noiseless <- t(sapply(cluster_true, function(cl) if (cl == 1) mu_1 else mu_2))
  
  set.seed(seed)
  x <- x_noiseless + matrix(rnorm(n * p), nrow = n, ncol = p)
  list(x = x, cluster_true = cluster_true)
}

generate_sparse_Graph_Gaussian <- function(sep, s, p, n,  rho, seed) {
  K <- 2
  cluster_true <- c(rep(1, n/2), rep(2, n/2))

  Omega <- diag(1, p)
  Omega[row(Omega) == col(Omega) - 1] <- rho  # lower diagonal
  Omega[row(Omega) == col(Omega) + 1] <- rho  # upper diagonal

  
  Sigma <- solve(Omega)# Covariance matrix Sigma
  M <- sep / 2 / sqrt(sum(Sigma[1:s, 1:s])) # Compute signal strength multiplier M

  # Sparse signal mean vector
  sparse_mean <- c(rep(1, s), rep(0, p - s))
  mu_0_tilde <- M * sparse_mean
  mu_0 <- Sigma %*% mu_0_tilde
  mu_1 <- -mu_0
  mu_2 <- mu_0
  x_noiseless <- t(sapply(cluster_true, function(cl) if (cl == 1) mu_1 else mu_2))
  
  # Compute beta for signal confirmation
  beta <- Omega %*% (mu_1 - mu_2)
  cat(sprintf("delta confirmed: %f\n", sqrt(t(mu_1 - mu_2) %*% beta)))
  norm_mu_diff <- sqrt(sum((mu_1 - mu_2)^2))
  cat(sprintf("norm(mu1 - mu2): %f\n", norm_mu_diff))
  L <- chol(Sigma)  # Cholesky decomposition (Sigma = L^T L)
  set.seed(seed)
    Z <- matrix(rnorm(n * p), nrow = n, ncol = p)
  noise <- Z %*% t(L)

  x <- x_noiseless + noise
  list(x = x, cluster_true = cluster_true)
}















