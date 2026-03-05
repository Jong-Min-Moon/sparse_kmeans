cat("\n\n=== Deriving B(alpha) ===\n")
# We know B(alpha) should equal C(alpha) * (alpha * gamma)
# Let's express B using X3, X4, X6 in terms of alpha, beta, gamma algebraically.
# We will use the 'Ryacas' package if possible, but let's just write evaluate loops to test guesses.

source("code_test/debug_algebra.R")
y <- alpha

# C(y) is 10*X3^3 - 7*X3*X4*y - 2*X3*y^3
# (alpha * gamma) = D / A
# So  B_true = C * (alpha * gamma_true)
cat("Target B_true:", C * (alpha * gamma_true), "\n")

# B(y) from paper is:
B_paper <- 4*X4_true^3 - 4*X3_true^2*X4_true*y - 8*X3_true^2*y^3 - X4_true^2*y^2 + 8*X4_true*y^4 + X6_true*y^3 + 4*y^6
cat("B_paper:", B_paper, "\n")

# Let's find the missing factor.
# What if it's X6_true * X3_true^2 ? No..
# Actually, the paper equations are solving a linear system to eliminate beta.
# X3 = alpha * B + 3 * (alpha*gamma)
# X4 = -2 alpha^2 + alpha * B^2 + 6 * B * (alpha*gamma) + 3 alpha * gamma^2
# In Moitra's code / paper, there's often a typo where X3 vs X3^2 is mismatched.
