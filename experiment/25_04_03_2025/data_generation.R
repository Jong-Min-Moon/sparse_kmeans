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