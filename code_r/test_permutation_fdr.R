# Test Permutation FDR
source("block_coordinate_optim_greedy_unknowncov_SAM.R")

# Mock dependencies if not loaded
# Assuming 'block_coordinate_optim_deterministic_unknowncov.R' and others are sourced in real env
# For this test, we need to mock/stub or assume they exist.
# Let's rely on the user's environment BUT define minimal mocks for the test to run independently if needed.

# Stub ESSC/ISEE if they don't exist in this raw script context
if (!exists("ESSC")) {
    ESSC <- function(X, K) {
        return(kmeans(t(X), K)$cluster)
    }
}
if (!exists("ISEE_residual_lasso")) {
    ISEE_residual_lasso <- function(X, cluster, K) {
        # Mock ISEE: Identity X_tilde, Ones Omega
        return(list(X_tilde = X, Omega_diag = rep(1, nrow(X))))
    }
}
if (!exists("get_cov_small")) {
    get_cov_small <- function(X, cl, s) {
        return(diag(1, length(s)))
    }
}
if (!exists("run_clustering_block_knowncov")) {
    run_clustering_block_knowncov <- function(X, s, K, cl, covariance) {
        return(list(cluster = cl)) # Mock: No change
    }
}
if (!exists("get_cluster_acc")) {
    get_cluster_acc <- function(a, b) {
        return(1.0)
    }
}

# 1. Generate Synthetic Data
set.seed(42)
p <- 100
n <- 50
K <- 2
X <- matrix(rnorm(p * n), nrow = p, ncol = n)

# Add signal to first 10 features
X[1:10, 1:(n / 2)] <- X[1:10, 1:(n / 2)] + 2 # Shift C1
X[1:10, (n / 2 + 1):n] <- X[1:10, (n / 2 + 1):n] - 2 # Shift C2

cat("Generated Data: 10 signal features, 90 noise features.\n")

# 2. Test get_permutation_fdr_threshold directly
cat("\n--- Testing Helper Function ---\n")
true_labels <- c(rep(1, n / 2), rep(2, n / 2))
# Mock stats: manually create logic
stats_obs <- rep(0, p)
stats_obs[1:10] <- 4 + rnorm(10) # Signal
stats_obs[11:100] <- abs(rnorm(90)) # Noise

cat("Mock Stats:\n Signal (Top 5): ", head(stats_obs[1:10]), "\n Noise (Top 5): ", head(stats_obs[11:100]), "\n")

res <- get_permutation_fdr_threshold(stats_obs, X, true_labels, rep(1, p), n_perms = 20, fdr_target = 0.1)

cat("Threshold Selected:", res$threshold, "\n")
cat("Features Selected:", res$n_selected, "\n")
cat("Estimated FDR:", res$est_fdr, "\n")

if (res$n_selected >= 10 && res$n_selected <= 15) {
    cat("SUCCESS: Selected around 10 signal features (some false pos allowed).\n")
} else {
    cat("WARNING: Selection count unexpected (Expected ~10-15).\n")
}

# 3. Test Full Function (Stubbed)
cat("\n--- Testing Full Function (Stubbed) ---\n")
tryCatch(
    {
        # We use a mocked run just to see if it crashes
        res_full <- block_coordinate_optim_greedy_unknowncov_SAM(X, K, n_iter = 2, n_perms = 5)
        cat("Function ran successfully.\n")
    },
    error = function(e) {
        cat("Error running function: ", e$message, "\n")
    }
)
