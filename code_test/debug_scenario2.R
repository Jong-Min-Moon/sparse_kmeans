source("code_r/hardt_price_gmm_1d.R")

cat("\n=== Testing Scenario 2 with Exact Analytical Moments ===\n")
p1 <- 0.25
p2 <- 0.75
mu1 <- 0.0
mu2 <- 3.0
sigma1 <- 0.5
sigma2 <- 4.0

M1_true <- p1 * mu1 + p2 * mu2
mu_shift <- M1_true

mu1_s <- mu1 - mu_shift
mu2_s <- mu2 - mu_shift

M2_true <- p1 * (mu1_s^2 + sigma1^2) + p2 * (mu2_s^2 + sigma2^2)
M3_true <- p1 * (mu1_s^3 + 3 * mu1_s * sigma1^2) + p2 * (mu2_s^3 + 3 * mu2_s * sigma2^2)
M4_true <- p1 * (mu1_s^4 + 6 * mu1_s^2 * sigma1^2 + 3 * sigma1^4) + p2 * (mu2_s^4 + 6 * mu2_s^2 * sigma2^2 + 3 * sigma2^4)
M5_true <- p1 * (mu1_s^5 + 10 * mu1_s^3 * sigma1^2 + 15 * mu1_s * sigma1^4) + p2 * (mu2_s^5 + 10 * mu2_s^3 * sigma2^2 + 15 * mu2_s * sigma2^4)
M6_true <- p1 * (mu1_s^6 + 15 * mu1_s^4 * sigma1^2 + 45 * mu1_s^2 * sigma1^4 + 15 * sigma1^6) + p2 * (mu2_s^6 + 15 * mu2_s^4 * sigma2^2 + 45 * mu2_s^2 * sigma2^4 + 15 * sigma2^6)

X3_true <- M3_true
X4_true <- M4_true - 3 * M2_true^2
X5_true <- M5_true - 10 * M3_true * M2_true
X6_true <- M6_true - 15 * M4_true * M2_true + 30 * M2_true^3

cat(sprintf("Exact X3: %.6f\n", X3_true))
cat(sprintf("Exact X4: %.6f\n", X4_true))
cat(sprintf("Exact X5: %.6f\n", X5_true))
cat(sprintf("Exact X6: %.6f\n", X6_true))

res <- RecoverFromMoments(M1_true, M2_true, X3_true, X4_true, X5_true, X6_true, epsilon=1e-8)

cat("\nRecovered Parameters:\n")
cat(sprintf("Comp A: p = %.2f, mu = %.2f, sigma = %.2f\n", res$comp1$p, res$comp1$mu, res$comp1$sigma))
cat(sprintf("Comp B: p = %.2f, mu = %.2f, sigma = %.2f\n", res$comp2$p, res$comp2$mu, res$comp2$sigma))
