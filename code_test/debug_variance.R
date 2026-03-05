source("code_r/hardt_price_gmm_1d.R")

p1 <- 0.3
p2 <- 0.7
mu1_orig <- -2.0
mu2_orig <- 5.0
sigma1 <- 0.5
sigma2 <- 1.5

mu_overall <- p1 * mu1_orig + p2 * mu2_orig
mu1_true <- mu1_orig - mu_overall
mu2_true <- mu2_orig - mu_overall

alpha_true <- -mu1_true * mu2_true
beta_true <- mu1_true + mu2_true
gamma_true <- (sigma2^2 - sigma1^2) / (mu2_true - mu1_true)

sigma_overall_sq <- p1*p2*(mu1_orig - mu2_orig)^2 + p1*sigma1^2 + p2*sigma2^2

cat("TRUE sigma1_sq calc:\n")
cat("p1*mu1^2 + p2*mu2^2 - mu1*gamma =", p1*mu1_true^2 + p2*mu2_true^2 - mu1_true*gamma_true, "\n")
cat("sigma2 - above =", sigma_overall_sq - (p1*mu1_true^2 + p2*mu2_true^2 - mu1_true*gamma_true), "\n")
cat("True sigma1^2 =", sigma1^2, "\n")

cat("\nRECOVERED:\n")
res <- RecoverFromMoments(
  mu = mu_overall,
  sigma2 = sigma_overall_sq,
  X3 = p1*mu1_true^3 + p2*mu2_true^3 + 3*gamma_true*alpha_true, # using true moments! Let's just pass exact X_i
  # Actually wait, let's just pass exact X_i to see if the equations are 100% correct.
  X4 = p1*mu1_true^4 + p2*mu2_true^4 + 6*p1*mu1_true^2*sigma1^2 + 6*p2*mu2_true^2*sigma2^2 + 3*p1*sigma1^4 + 3*p2*sigma2^4 - 3*sigma_overall_sq^2,
  X5 = p1*mu1_true^5 + p2*mu2_true^5, # This is wrong, X5 involves M5 etc... let's just compute M_i
  X6 = 0,
  epsilon = 1e-4
)

EOF
