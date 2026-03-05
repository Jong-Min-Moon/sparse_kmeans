source("code_test/debug_algebra.R")

y <- alpha
p6_corrected <- (4 * X3_true^2 - 3 * X4_true * y - 2 * y^3) * 
      (4 * X3_true^4 - 4 * X3_true^2 * X4_true * y - 8 * X3_true^2 * y^3 - X4_true^2 * y^2 + 8 * X4_true * y^4 + X6_true * y^3 + 4 * y^6) - 
      (10 * X3_true^3 - 7 * X3_true * X4_true * y - 2 * X3_true * y^3) * 
      (2 * X3_true^3 - 3 * X3_true * X4_true * y + 2 * X3_true * y^3 + X5_true * y^2)

cat("\nCorrected p6:", p6_corrected, "\n")
