source("code_test/debug_scenario2.R")
source("code_r/hardt_price_gmm_1d.R")

true_alpha <- 1.6875
cat(sprintf("\nTrue Alpha: %.4f\n", true_alpha))

roots <- polyroot(c(-X3_true^2, X4_true, 0, 2))
real_roots <- Re(roots)[abs(Im(roots)) < 1e-8]
ymax <- max(real_roots)
cat(sprintf("ymax: %.4f\n", ymax))

kappa <- 1 + sqrt(abs(X4_true))/ymax
epsilon <- 1e-8
upper_limit <- (1 + epsilon/kappa) * ymax
cat(sprintf("Upper Limit: %.4f\n", upper_limit))

cat(sprintf("p5(true_alpha) = %.6f\n", sqrt(evaluate_r(true_alpha, X3_true, X4_true, X5_true, X6_true))))

r_val_true <- evaluate_r(true_alpha, X3_true, X4_true, X5_true, X6_true)
threshold <- (epsilon^2) * (true_alpha^18) * (kappa^10)
cat(sprintf("r(true_alpha) = %.12e\n", r_val_true))
cat(sprintf("Threshold   = %.12e\n", threshold))

cand <- RecoverAlphaFromMoments(X3_true, X4_true, X5_true, X6_true, epsilon)
cat(sprintf("Recovered Alpha: %.4f\n", cand))
