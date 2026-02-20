# Validation Script for ISEE_bicluster.R
# rigorous mathematical audit: Ground Truth vs Estimated Transformation

library(MASS)
library(ggplot2)
library(foreach)
library(reshape2)

# Source functions
if (file.exists("code_r/ISEE_bicluster.R")) {
    source("code_r/ISEE_bicluster.R")
    source("code_r/get_intercept_residual_lasso.R")
} else {
    stop("Please run this script from the project root (where code_r is located).")
}

# 1. Ground Truth Definition
cat("1. Generating Ground Truth Data...\n")
set.seed(2025)

p <- 200
n <- 500
n_c <- n / 2

# Sparse Precision Matrix (Omega) - Tridiagonal
Omega_true <- matrix(0, p, p)
for (i in 1:p) {
    Omega_true[i, i] <- 1
    if (i > 1) Omega_true[i, i - 1] <- 0.5
    if (i < p) Omega_true[i, i + 1] <- 0.5
}
# Ensure positive definite
if (min(eigen(Omega_true)$values) <= 0) {
    Omega_true <- Omega_true + diag(0.1, p)
}

Sigma <- solve(Omega_true)

# Means
mu1 <- rep(2, p)
mu2 <- rep(-2, p)

# Generate Data X (p x n)
X1 <- mvrnorm(n_c, mu1, Sigma) # n_c x p
X2 <- mvrnorm(n_c, mu2, Sigma) # n_c x p
X <- t(rbind(X1, X2)) # p x n

true_labels <- c(rep(1, n_c), rep(2, n_c))

# Perturb Cluster Assignments
perturb_rate <- 0
n_perturb <- floor(n * perturb_rate)
cat(sprintf("Perturbing %d labels (%.0f%%)...\n", n_perturb, perturb_rate * 100))

est_labels <- true_labels
perturb_idx <- sample(n, n_perturb)
# Flip labels (assuming 1 and 2)
est_labels[perturb_idx] <- 3 - est_labels[perturb_idx]

# Target Matrix (Ground Truth Transformed X)
# X_tilde_truth = Omega * X (non-centered)
X_tilde_truth <- Omega_true %*% X

# 2. Run ISEE_bicluster
cat("2. Running ISEE_bicluster with perturbed labels...\n")

# Register parallel backend if not already
if (getDoParWorkers() == 1) {
    if (requireNamespace("doParallel", quietly = TRUE)) {
        doParallel::registerDoParallel(cores = 2)
    } else {
        registerDoSEQ()
    }
}

# res is now just X_tilde (matrix)
X_tilde_ISEE <- ISEE_bicluster(X, est_labels)

# 3. Metric Overhaul

cat("\n3. Evaluation Metrics:\n")

# A. Normalized Frobenius Norm (Relative Error)
# || A - B ||_F / || B ||_F
frobenius_diff <- norm(X_tilde_ISEE - X_tilde_truth, type = "F")
frobenius_true <- norm(X_tilde_truth, type = "F")
relative_error <- frobenius_diff / frobenius_true

cat(sprintf("   Normalized Frobenius Error: %.5f\n", relative_error))

# B. Direct Transformation Recovery (Mean Column Correlation)
# Compute correlation for each column (sample) and take mean
cor_vec <- numeric(n)
for (i in 1:n) {
    cor_vec[i] <- cor(X_tilde_ISEE[, i], X_tilde_truth[, i])
}
mean_col_cor <- mean(cor_vec)
cat(sprintf("   Mean Column Correlation:    %.5f\n", mean_col_cor))

# C. Precision Diagonal Accuracy (MAE) - SKIPPED
cat("   Precision Diagonal MAE:     N/A (Not returned by modified ISEE_bicluster)\n")

# 4. Visual Diagnostics
cat("\n4. Creating Diagnostic Plots...\n")

# A. Identity Check Plot
# Scatter plot of Truth vs Estimated elements
df_identity <- data.frame(
    Truth = as.vector(X_tilde_truth),
    Estimated = as.vector(X_tilde_ISEE)
)

p1 <- ggplot(df_identity, aes(x = Truth, y = Estimated)) +
    geom_point(alpha = 0.3, color = "blue") +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    ggtitle("Identity Check: Truth vs Estimated X_tilde") +
    labs(x = "True Transformed Value (Omega * X)", y = "Estimated Value (ISEE)") +
    theme_minimal() +
    coord_fixed()

ggsave("validation_identity_check.png", p1, width = 6, height = 6)

cat("Validation complete. Plots saved:\n")
cat("  - validation_identity_check.png\n")
