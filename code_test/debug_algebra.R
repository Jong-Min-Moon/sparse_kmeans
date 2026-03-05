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

alpha <- -mu1 * mu2
beta <- mu1 + mu2
gamma <- (sigma2^2 - sigma1^2) / (mu2 - mu1)

cat("Analytical X3:", X3_true, "\n")
cat("Paper Eq X3:", alpha*beta + 3*alpha*gamma, "\n\n")

cat("Analytical X4:", X4_true, "\n")
cat("Paper Eq X4:", -2*alpha^2 + alpha*beta^2 + 6*alpha*beta*gamma + 3*alpha*gamma^2, "\n\n")

cat("Analytical X5:", X5_true, "\n")
cat("Paper Eq X5:", alpha*(beta^3 - 8*alpha*beta + 10*beta^2*gamma + 15*gamma^2*beta - 20*alpha*gamma), "\n\n")

cat("Analytical X6:", X6_true, "\n")
cat("Paper Eq X6:", alpha*(16*alpha^2 - 12*alpha*beta^2 - 60*alpha*beta*gamma + beta^4 + 15*beta^3*gamma + 45*beta^2*gamma^2 + 15*beta*gamma^3), "\n\n")

gamma_num <- alpha^2 * X5_true + 2 * X3_true^3 + 2 * alpha^3 * X3_true - 3 * X3_true * X4_true * alpha
gamma_den <- 4 * X3_true^2 - 2 * alpha^3 - 3 * X4_true * alpha
gamma_computed <- (1 / alpha) * (gamma_num / gamma_den)

cat("Computed gamma_num:", gamma_num, "\n")
cat("Computed gamma_den:", gamma_den, "\n")
cat("Computed gamma:", gamma_computed, " (True:", gamma, ")\n")

beta_computed <- (1 / alpha) * (X3_true - 3 * alpha * gamma_computed)
cat("Computed beta:", beta_computed, " (True:", beta, ")\n")
