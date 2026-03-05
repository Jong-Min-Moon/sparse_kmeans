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

M2_true <- p1 * (mu1^2 + sigma1^2) + p2 * (mu2^2 + sigma2^2)
M3_true <- p1 * (mu1^3 + 3*mu1*sigma1^2) + p2 * (mu2^3 + 3*mu2*sigma2^2)
M4_true <- p1 * (mu1^4 + 6*mu1^2*sigma1^2 + 3*sigma1^4) + p2 * (mu2^4 + 6*mu2^2*sigma2^2 + 3*sigma2^4)
M5_true <- p1 * (mu1^5 + 10*mu1^3*sigma1^2 + 15*mu1*sigma1^4) + p2 * (mu2^5 + 10*mu2^3*sigma2^2 + 15*mu2*sigma2^4)
M6_true <- p1 * (mu1^6 + 15*mu1^4*sigma1^2 + 45*mu1^2*sigma1^4 + 15*sigma1^6) + p2 * (mu2^6 + 15*mu2^4*sigma2^2 + 45*mu2^2*sigma2^4 + 15*sigma2^6)

X3_true <- M3_true
X4_true <- M4_true - 3 * M2_true^2
X5_true <- M5_true - 10 * M3_true * M2_true
X6_true <- M6_true - 15 * M4_true * M2_true + 30 * M2_true^3

cat("\nRecovering from EXACT TRUE analytical moments...\n")
res <- RecoverFromMoments(
  mu = mu_overall,
  sigma2 = M2_true,
  X3 = X3_true,
  X4 = X4_true,
  X5 = X5_true,
  X6 = X6_true,
  epsilon = 1e-4
)

cat(sprintf("Recovered Comp 1: p=%.2f, mu=%.2f, sigma=%.2f (True: %.2f)\n", res$comp1$p, res$comp1$mu, res$comp1$sigma, sigma1))
cat(sprintf("Recovered Comp 2: p=%.2f, mu=%.2f, sigma=%.2f (True: %.2f)\n", res$comp2$p, res$comp2$mu, res$comp2$sigma, sigma2))
cat(sprintf("Recovered Alpha: %.4f (True: %.4f)\n", res$alpha, -mu1 * mu2))

