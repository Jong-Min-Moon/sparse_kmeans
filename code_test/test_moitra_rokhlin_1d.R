source("code_r/moitra_rokhlin_gmm_1d.R")

# True parameters
p1_true <- 0.3
p2_true <- 0.7
mu1_true <- 5
mu2_true <- -2
sigma1_true <- 1.5
sigma2_true <- 0.5

set.seed(123)
n <- 500000

# Generate data
labels <- sample(1:2, n, replace = TRUE, prob = c(p1_true, p2_true))
x <- numeric(n)
n1 <- sum(labels == 1)
n2 <- n - n1
x[labels == 1] <- rnorm(n1, mean = mu1_true, sd = sigma1_true)
x[labels == 2] <- rnorm(n2, mean = mu2_true, sd = sigma2_true)

cat("True Parameters:\n")
cat(sprintf("Comp 1: p = %.2f, mu = %.2f, sigma = %.2f\n", p1_true, mu1_true, sigma1_true))
cat(sprintf("Comp 2: p = %.2f, mu = %.2f, sigma = %.2f\n", p2_true, mu2_true, sigma2_true))
cat("----------------------------------\n")

# Estimate moments
moments <- estimate_excess_moments(x)

# Recover parameters using Moitra & Rokhlin Algorithm 3.1
res <- RecoverFromMoments(
  mu = moments$mu,
  sigma2 = moments$sigma2,
  X3 = moments$X3,
  X4 = moments$X4,
  X5 = moments$X5,
  X6 = moments$X6,
  epsilon = 1e-4
)

cat("Recovered Parameters:\n")
cat(sprintf("Comp 1: p = %.2f, mu = %.2f, sigma = %.2f\n", res$comp1$p, res$comp1$mu, res$comp1$sigma))
cat(sprintf("Comp 2: p = %.2f, mu = %.2f, sigma = %.2f\n", res$comp2$p, res$comp2$mu, res$comp2$sigma))

# Wait, the algorithm output might map comp1 to true comp2. Let's sort by mu.
