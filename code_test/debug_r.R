source("code_r/hardt_price_gmm_1d.R")
source("code_test/debug_algebra.R")

cat("\n=== Debugging p5 and p6 at TRUE ALPHA ===\n")
y <- alpha

p5 <- 6 * (2 * X3_true * y^3 + X5_true * y^2 - 3 * X3_true * X4_true * y + 2 * X3_true^3)^2 + 
      (2 * y^3 + 3 * X4_true * y - 4 * X3_true^2)^2 * (2 * y^3 + X4_true * y - X3_true^2)
      
p6 <- (4 * X3_true^2 - 3 * X4_true * y - 2 * y^3) * 
      (4 * X4_true^3 - 4 * X3_true^2 * X4_true * y - 8 * X3_true^2 * y^3 - X4_true^2 * y^2 + 8 * X4_true * y^4 + X6_true * y^3 + 4 * y^6) - 
      (10 * X3_true^3 - 7 * X3_true * X4_true * y - 2 * X3_true * y^3) * 
      (2 * X3_true^3 - 3 * X3_true * X4_true * y + 2 * X3_true * y^3 + X5_true * y^2)

cat("p5 at alpha_true:", p5, "\n")
cat("p6 at alpha_true:", p6, "\n")
cat("r(alpha_true):", p5^2 + p6^2, "\n")

# Why did optimizer choose 10.4022 instead?
grid <- seq(10.2, 10.5, length.out=100)
r_vals <- evaluate_r(grid, X3_true, X4_true, X5_true, X6_true)
min_idx <- which.min(r_vals)
cat("\nGrid minimum at:", grid[min_idx], "with r(val) =", r_vals[min_idx], "\n")
