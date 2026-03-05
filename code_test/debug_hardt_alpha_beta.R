source("code_r/hardt_price_gmm_1d.R")

p1 <- 0.3
p2 <- 0.7
mu1_orig <- -2.0
mu2_orig <- 5.0
sigma1 <- 0.5
sigma2 <- 1.5

mu_overall <- p1 * mu1_orig + p2 * mu2_orig
mu1 <- mu1_orig - mu_overall
mu2 <- mu2_orig - mu_overall

cat("True shifted means:", mu1, mu2, "\n")
alpha_true <- -mu1 * mu2
beta_true <- mu1 + mu2
gamma_true <- (sigma2^2 - sigma1^2) / (mu2 - mu1)

cat(sprintf("True alpha: %.4f, beta: %.4f, gamma: %.4f\n", alpha_true, beta_true, gamma_true))

# Generate dataset to trace execution values
set.seed(42)
n <- 5000000
labels <- sample(1:2, n, replace = TRUE, prob = c(p1, p2))
x <- numeric(n)
n1 <- sum(labels == 1)
n2 <- n - n1
x[labels == 1] <- rnorm(n1, mean = mu1_orig, sd = sigma1)
x[labels == 2] <- rnorm(n2, mean = mu2_orig, sd = sigma2)

moments <- estimate_excess_moments(x)
cat("\nRecovering from moments...\n")
res <- RecoverFromMoments(
  mu = moments$mu,
  sigma2 = moments$sigma2,
  X3 = moments$X3,
  X4 = moments$X4,
  X5 = moments$X5,
  X6 = moments$X6,
  epsilon = 1e-4
)

cat(sprintf("Recovered alpha: %.4f, beta: %.4f, gamma: %.4f\n", res$alpha, res$beta, res$gamma))
