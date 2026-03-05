source("code_test/debug_scenario2.R")

cat("\n=== Testing Symbolic Polynomial Expansion for Jenkins-Traub ===\n")

poly_add <- function(p1, p2) {
  n <- max(length(p1), length(p2))
  p1 <- c(p1, rep(0, n - length(p1)))
  p2 <- c(p2, rep(0, n - length(p2)))
  return(p1 + p2)
}

poly_mul <- function(p1, p2) {
  res <- numeric(length(p1) + length(p2) - 1)
  for (i in seq_along(p1)) {
    for (j in seq_along(p2)) {
      res[i + j - 1] <- res[i + j - 1] + p1[i] * p2[j]
    }
  }
  return(res)
}

# p5 has the form:
# 6 * (2 * X3 * y^3 + X5 * y^2 - 3 * X3 * X4 * y + 2 * X3^3)^2 + 
# (2 * y^3 + 3 * X4 * y - 4 * X3^2)^2 * (2 * y^3 + X4 * y - X3^2)

# term 1: (2*X3^3  - 3*X3*X4 * y  + X5 * y^2 + 2*X3 * y^3)
t1_base <- c(2 * X3_true^3, -3 * X3_true * X4_true, X5_true, 2 * X3_true)
t1_sq <- poly_mul(t1_base, t1_base)
term1 <- t1_sq * 6

# term 2: (-4*X3^2 + 3*X4 * y + 0*y^2 + 2 * y^3)^2 * (-X3^2 + X4*y + 0*y^2 + 2*y^3)
t2_base1 <- c(-4 * X3_true^2, 3 * X4_true, 0, 2)
t2_sq <- poly_mul(t2_base1, t2_base1)

t2_base2 <- c(-X3_true^2, X4_true, 0, 2)
term2 <- poly_mul(t2_sq, t2_base2)

p5_coeffs <- poly_add(term1, term2)

cat("p5 coefficients:\n")
print(p5_coeffs)

all_roots <- polyroot(p5_coeffs)
real_roots <- Re(all_roots)[abs(Im(all_roots)) < 1e-8]
real_roots <- real_roots[real_roots > 0]
cat("\nReal Roots > 0 Found via Jenkins-Traub:\n")
print(real_roots)

cat("\nChecking with ymax and bounds:\n")
p_ymax <- c(-X3_true^2, X4_true, 0, 2)
ymax_roots <- Re(polyroot(p_ymax))[abs(Im(polyroot(p_ymax))) < 1e-8]
ymax <- max(ymax_roots)
kappa <- 1 + sqrt(abs(X4_true)) / ymax
upper_limit <- (1 + 1e-6 / kappa) * ymax

cat(sprintf("Upper limit: %.6f\n", upper_limit))
valid_roots <- real_roots[real_roots <= upper_limit + 1e-6]
cat("Valid Candidates:\n")
print(valid_roots)
