source("code_r/moitra_rokhlin_gmm_1d.R")

# True parameters (same mean, different variances)
p1_true <- 0.6
p2_true <- 0.4
mu_true <- 3.5
sigma1_true <- 4.0
sigma2_true <- 1.0

set.seed(42)
n <- 500000

# Generate data
labels <- sample(1:2, n, replace = TRUE, prob = c(p1_true, p2_true))
x <- numeric(n)
n1 <- sum(labels == 1)
n2 <- n - n1
x[labels == 1] <- rnorm(n1, mean = mu_true, sd = sigma1_true)
x[labels == 2] <- rnorm(n2, mean = mu_true, sd = sigma2_true)

cat("True Parameters:\n")
cat(sprintf("Comp 1: p = %.2f, mu = %.2f, sigma = %.2f\n", p1_true, mu_true, sigma1_true))
cat(sprintf("Comp 2: p = %.2f, mu = %.2f, sigma = %.2f\n", p2_true, mu_true, sigma2_true))
cat("----------------------------------\n")

# Estimate moments
moments <- estimate_excess_moments(x)

# Recover parameters using Moitra & Rokhlin Algorithm 3.2
res <- SameMeanRecoverFromMoments(
  mu = moments$mu,
  sigma2 = moments$sigma2,
  X4 = moments$X4,
  X6 = moments$X6
)

cat("Recovered Parameters:\n")
cat(sprintf("Comp 1: p = %.2f, mu = %.2f, sigma = %.2f\n", res$comp1$p, res$comp1$mu, res$comp1$sigma))
cat(sprintf("Comp 2: p = %.2f, mu = %.2f, sigma = %.2f\n", res$comp2$p, res$comp2$mu, res$comp2$sigma))
