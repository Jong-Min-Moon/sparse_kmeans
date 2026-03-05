source("code_r/hardt_price_gmm_1d.R")
source("code_test/debug_algebra.R")

cat("\n=== Testing GMM Parameters for p5(y) Roots ===\n")
y_candidates <- c(2.264, 10.29) # Roughly the two roots

for (alpha_cand in y_candidates) {
  cat(sprintf("\n--- Testing candidate alpha = %.4f ---\n", alpha_cand))
  
  gamma_num <- alpha_cand^2 * X5_true + 2 * X3_true^3 + 2 * alpha_cand^3 * X3_true - 3 * X3_true * X4_true * alpha_cand
  gamma_den <- 4 * X3_true^2 - 2 * alpha_cand^3 - 3 * X4_true * alpha_cand
  gamma_cand <- (1 / alpha_cand) * (gamma_num / gamma_den)
  
  beta_cand <- (1 / alpha_cand) * (X3_true - 3 * alpha_cand * gamma_cand)
  
  disc <- beta_cand^2 + 4 * alpha_cand
  cat(sprintf("Discriminant: %.4f\n", disc))
  
  if (disc < 0) {
      cat("Invalid: Discriminant < 0\n")
      next
  }
  
  disc <- sqrt(disc)
  mu1 <- (beta_cand - disc) / 2
  mu2 <- (beta_cand + disc) / 2
  
  p1 <- mu2 / (mu2 - mu1)
  p2 <- -mu1 / (mu2 - mu1)
  
  cat(sprintf("p1: %.4f, p2: %.4f\n", p1, p2))
  if (p1 < 0 || p1 > 1) {
      cat("Invalid: Probabilities out of bounds\n")
      next
  }
  
  sigma1_sq <- M2_true - (p1 * mu1^2 + p2 * mu2^2 - mu1 * gamma_cand)
  sigma2_sq <- sigma1_sq + (mu2 - mu1) * gamma_cand
  
  cat(sprintf("sigma1_sq: %.4f, sigma2_sq: %.4f\n", sigma1_sq, sigma2_sq))
  if (sigma1_sq < 0 || sigma2_sq < 0) {
      cat("Invalid: Variances < 0\n")
      next
  }
  
  cat("=> ALL VALID! Candidate GMM is mathematically possible.\n")
}
