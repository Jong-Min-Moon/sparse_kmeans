source("code_test/debug_scenario2.R")
source("code_r/hardt_price_gmm_1d.R")

cat("\n=== Testing Candidate Precision ===\n")
cand <- 1.6875001
r_val <- evaluate_r(cand, X3_true, X4_true, X5_true, X6_true)
cat(sprintf("r at cand=%.8f is %f\n", cand, r_val))

cand2 <- 1.68750001
r_val2 <- evaluate_r(cand2, X3_true, X4_true, X5_true, X6_true)
cat(sprintf("r at cand=%.8f is %f\n", cand2, r_val2))

# What happens if we refine cand exactly via polyroot?
roots <- polyroot(c(-X3_true^2, X4_true, 0, 2))
real_roots <- Re(roots)[abs(Im(roots)) < 1e-8]
ymax <- max(real_roots)

kappa <- 1 + sqrt(abs(X4_true))/ymax
epsilon <- 1e-8
threshold <- (epsilon^2) * (1.6875^18) * (kappa^10)
cat(sprintf("\nThreshold is: %.12e\n", threshold))

