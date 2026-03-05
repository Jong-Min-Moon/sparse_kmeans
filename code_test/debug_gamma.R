source("code_r/hardt_price_gmm_1d.R")
source("code_test/debug_algebra.R")

cat("\n\n=== Debugging Gamma & Beta Reconstructions ===\n")

gamma_num <- alpha^2 * X5_true + 2 * X3_true^3 + 2 * alpha^3 * X3_true - 3 * X3_true * X4_true * alpha
gamma_den <- 4 * X3_true^2 - 2 * alpha^3 - 3 * X4_true * alpha
gamma_computed <- (1 / alpha) * (gamma_num / gamma_den)

cat("Computed gamma_num:", gamma_num, "\n")
cat("Computed gamma_den:", gamma_den, "\n")
cat("Computed gamma:", gamma_computed, " (True:", gamma_true, ")\n")

beta_computed <- (1 / alpha) * (X3_true - 3 * alpha * gamma_computed)
cat("Computed beta:", beta_computed, " (True:", beta_true, ")\n")
