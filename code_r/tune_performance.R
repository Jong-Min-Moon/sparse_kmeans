library(stats)
library(mclust)

# Source ADMM solver
if (file.exists("code_r/sdp_kmeans.R")) {
    source("code_r/sdp_kmeans.R")
} else {
    source("../../code_r/sdp_kmeans.R")
}

# Parameters
n <- 200
K <- 2
s <- 10
p <- 3000
mu_val <- sqrt(4 / s)

cat("Tuning ADMM Parameters\n")

set.seed(4001)
n1 <- n / 2
n2 <- n / 2
mu1 <- numeric(p)
mu2 <- numeric(p)
mu1[1:s] <- mu_val
mu2[1:s] <- -mu_val

X1 <- matrix(rnorm(n1 * p), nrow = p) + mu1
X2 <- matrix(rnorm(n2 * p), nrow = p) + mu2
X <- cbind(X1, X2)
G <- crossprod(X)
true_labels <- c(rep(1, n1), rep(2, n2))

# Run with default parameters
cat("\n--- Default Parameters (mu=10, tau=2) ---\n")
t_start <- Sys.time()
res_def <- sdp_kmeans(G, K, max_iter = 2000, tol = 1e-4, verbose = FALSE, k_prime_factor = 3, mu = 10.0, tau = 2.0)
t_end <- Sys.time()
acc_def <- max(mean(res_def$cluster == true_labels), mean(res_def$cluster != true_labels))
cat(sprintf("Time: %.2fs, Acc: %.2f, Iter: %d\n", as.numeric(t_end - t_start, units = "secs"), acc_def, res_def$iter))

# Run with tuned parameters (Relaxed mu, slower tau)
# mu = 2.0 means we trigger update if one residual is > 2x the other (more aggressive adaptation)
# tau = 1.5 means we change rho by 1.5x (smoother change)
cat("\n--- Tuned Parameters (mu=2, tau=1.5) ---\n")
t_start <- Sys.time()
res_tune <- sdp_kmeans(G, K, max_iter = 2000, tol = 1e-4, verbose = FALSE, k_prime_factor = 3, mu = 2.0, tau = 1.5)
t_end <- Sys.time()
acc_tune <- max(mean(res_tune$cluster == true_labels), mean(res_tune$cluster != true_labels))
cat(sprintf("Time: %.2fs, Acc: %.2f, Iter: %d\n", as.numeric(t_end - t_start, units = "secs"), acc_tune, res_tune$iter))
