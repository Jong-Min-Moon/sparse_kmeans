library(MASS)
source("code_r/hardt_price_gmm_1d.R")
source("code_r/hardt_price_gmm_nd.R")
source("code_r/hardt_price_alg_B.R")

cat("=======================================================\n")
cat("=== Testing Alg B (d-to-4 Reduction) on d=6 Mixture ===\n")
cat("=======================================================\n")

set.seed(42)
d <- 6
n <- 20000

# Ground Truth Parameters
mu1 <- c(-2, -2, 2, 2, 0, 0)
mu2 <- c(2, 2, -2, -2, 4, 4)

S1 <- diag(1, d)
S2 <- diag(2, d)

# Inject dense correlations into S2
S2[1,2] <- S2[2,1] <- 0.8
S2[3,4] <- S2[4,3] <- -0.5
S2[5,6] <- S2[6,5] <- 1.2

# Ensure PSD
eig2 <- eigen(S2)$values
if (any(eig2 < 0)) stop("S2 is not PSD")

# Generate Data
X1 <- mvrnorm(n * 0.5, mu1, S1)
X2 <- mvrnorm(n * 0.5, mu2, S2)
X <- rbind(X1, X2)

# Run Recovery!
cat("\n[TEST] Executing ReduceDTo4...\n")
res <- ReduceDTo4(X, epsilon = 0.5, delta = 0.05)

cat("\n=== ground truth ===\n")
cat("mu1: ", mu1, "\n")
cat("mu2: ", mu2, "\n\n")

cat("=== Recovered Means ===\n")
cat("rec_mu A: ", round(res$comp1$mu, 3), "\n")
cat("rec_mu B: ", round(res$comp2$mu, 3), "\n\n")

# Align components based on distance to mu1
distA <- sum((res$comp1$mu - mu1)^2)
distB <- sum((res$comp2$mu - mu1)^2)

if (distA < distB) {
  mu_err_1 <- max(abs(res$comp1$mu - mu1))
  mu_err_2 <- max(abs(res$comp2$mu - mu2))
  
  sig_err_1 <- sqrt(sum((res$comp1$sigma - S1)^2))
  sig_err_2 <- sqrt(sum((res$comp2$sigma - S2)^2))
} else {
  mu_err_1 <- max(abs(res$comp2$mu - mu1))
  mu_err_2 <- max(abs(res$comp1$mu - mu2))
  
  sig_err_1 <- sqrt(sum((res$comp2$sigma - S1)^2))
  sig_err_2 <- sqrt(sum((res$comp1$sigma - S2)^2))
}

cat(sprintf("Max Coordinate Mean Error (L-inf): %.4f\n", max(mu_err_1, mu_err_2)))
cat(sprintf("Max Covariance Error (Frobenius): %.4f\n", max(sig_err_1, sig_err_2)))

cat("\n[TEST] Summary: Recovery Complete.\n")
