source("code_test/debug_algebra.R")

cat("\n=== Testing exact paper equations for mu, p, sigma ===\n")

alpha_val <- alpha
beta_val <- -2.8 # true beta is -2.8
gamma_val <- 0.2857143 # true gamma is 2/7

# Paper equations exactly:
disc <- sqrt(beta_val^2 + 4*alpha_val)
# Test with +beta
mu1 <- (beta_val - disc)/2
mu2 <- (beta_val + disc)/2

p1 <- mu2 / (mu2 - mu1)
p2 <- -mu1 / (mu2 - mu1)

# Note: true mu1_orig was -2.0, true p1 was 0.3, true sigma1=0.5
# true mu2_orig was 5.0, true p2 was 0.7, true sigma2=1.5
# shifted true: mu1=-4.9, mu2=2.1

cat(sprintf("mu1: %.4f, mu2: %.4f\n", mu1, mu2))
cat(sprintf("p1: %.4f, p2: %.4f\n", p1, p2))

sigma_overall_sq <- 0.3 * 0.7 * (-2.0 - 5.0)^2 + 0.3 * 0.5^2 + 0.7 * 1.5^2

sigma1_sq <- sigma_overall_sq - (p1*mu1^2 + p2*mu2^2 - mu1*gamma_val)
sigma2_sq <- sigma1_sq + (mu2 - mu1)*gamma_val

cat(sprintf("sigma1_sq: %.4f (True: %.4f)\n", sigma1_sq, 0.5^2))
cat(sprintf("sigma2_sq: %.4f (True: %.4f)\n", sigma2_sq, 1.5^2))
