source("code_r/hardt_price_gmm_1d.R")
source("code_test/debug_algebra.R")

cat("\n=== Checking p5(y) Roots on (0, ymax] ===\n")
roots <- polyroot(c(-X3_true^2, X4_true, 0, 2))
real_roots <- Re(roots)[abs(Im(roots)) < 1e-8]
ymax <- max(real_roots)

cat("ymax:", ymax, "\n")
grid <- seq(1e-4, ymax, length.out=1000)

p5_vals <- sapply(grid, function(y) {
  6 * (2 * X3_true * y^3 + X5_true * y^2 - 3 * X3_true * X4_true * y + 2 * X3_true^3)^2 + 
  (2 * y^3 + 3 * X4_true * y - 4 * X3_true^2)^2 * (2 * y^3 + X4_true * y - X3_true^2)
})

# Find roots by looking for zero crossings
signs <- sign(p5_vals)
crossings <- which(diff(signs) != 0)

cat("\np5(y) crosses 0 at roughly:\n")
for (i in crossings) {
    cat(sprintf("Grid interval [%.4f, %.4f]\n", grid[i], grid[i+1]))
}

cat("True alpha:", alpha, "\n")
