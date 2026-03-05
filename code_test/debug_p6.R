source("code_r/hardt_price_gmm_1d.R")
source("code_test/debug_algebra.R")

cat("\n=== Debugging D, E, F at TRUE ALPHA ===\n")
y <- alpha

D <- 2 * X3_true * y^3 + X5_true * y^2 - 3 * X3_true * X4_true * y + 2 * X3_true^3
E <- 2 * y^3 + 3 * X4_true * y - 4 * X3_true^2
F_val <- 2 * y^3 + X4_true * y - X3_true^2

A <- 4 * X3_true^2 - 3 * X4_true * y - 2 * y^3
B <- 4 * X4_true^3 - 4 * X3_true^2 * X4_true * y - 8 * X3_true^2 * y^3 - X4_true^2 * y^2 + 8 * X4_true * y^4 + X6_true * y^3 + 4 * y^6
C <- 10 * X3_true^3 - 7 * X3_true * X4_true * y - 2 * X3_true * y^3

cat("D:", D, "\n")
cat("E:", E, "\n")
cat("F:", F_val, "\n")
cat("A:", A, "\n")
cat("B:", B, "\n")
cat("C:", C, "\n")
cat("A*B:", A*B, "\n")
cat("C*D:", C*D, "\n")
