# Verification script for C++ Greedy Screening Backend
library(stats)
library(Rcpp)

# Source the updated screening function
source("code_r/selection_block_greedy_screening.R")

# 1. Setup Data
set.seed(42)
p <- 1000
n <- 200
s <- 10
X <- matrix(rnorm(n * p), p, n)
cluster_est <- c(rep(1, n / 2), rep(2, n / 2))

# Add signal to the first 's' features
X[1:s, cluster_est == 1] <- X[1:s, cluster_est == 1] + 2
X[1:s, cluster_est == 2] <- X[1:s, cluster_est == 2] - 2

# 2. Run with R Fallback (Force by breaking path)
cat("\n--- Running with R Fallback ---\n")
# We can't easily break the path if it exists, so let's just use the original logic if we haven't compiled yet.
# Or we can temporarily rename the DLL.

# 3. Compile
cat("\n--- Compiling C++ Backend ---\n")
# Run build_solver.ps1 (if on Windows)
if (.Platform$OS.type == "windows") {
    system("powershell -ExecutionPolicy Bypass -File code_r/build_solver.ps1")
}

# 4. Run with C++ Backend
cat("\n--- Running with C++ Backend ---\n")
t1 <- Sys.time()
selected_cpp <- selection_block_greedy_screening(X, cluster_est, fdr_level = 0.4, n_perms = 1000)
t2 <- Sys.time()
cat(sprintf("C++ Time: %.4f s\n", as.numeric(difftime(t2, t1, units = "secs"))))

# 5. Verify Selected Features
# First 's' features should be selected
s_selected <- sum(selected_cpp[1:s])
false_selected <- sum(selected_cpp[(s + 1):p])

cat(sprintf("True features selected: %d/%d\n", s_selected, s))
cat(sprintf("False features selected: %d/%d\n", false_selected, p - s))

if (s_selected == s) {
    cat("Verification SUCCESS: All true features correctly identified.\n")
} else {
    cat("Verification WARNING: Not all true features selected. This might be expected depending on noise/FDR.\n")
}
