# ------------------------------------------------------------------
# toy_scvx_user.R
# Verification script for the SCVX grid search pipeline.
# Uses run_scvx() directly from code_r/scvx_wrapper.R.
# ------------------------------------------------------------------
source("../../../code_r/scvx_wrapper.R")
source("../../../code_r/get_cluster_acc.R")

library(scvxclustr)
library(cvxclustr)
library(igraph)
library(Matrix)
library(cluster)

# High-dimensional sparse simulator
simu_highdim <- function(n, true_p, p, mu, sigma) {
    n_per_cluster <- floor(n / 2)
    X      <- matrix(rnorm(n * p, sd = sigma), n, p)
    labels <- rep(1:2, each = n_per_cluster)
    if (length(labels) < n) labels <- c(labels, 2)
    X[labels == 1, 1:true_p] <- X[labels == 1, 1:true_p] + mu
    X[labels == 2, 1:true_p] <- X[labels == 2, 1:true_p] - mu
    return(list(X = X, label = labels, features = 1:true_p))
}

n_val     <- 40
true_p    <- 20
p_val     <- 2000
mu_val    <- 4
sigma_val <- 1

cat("Generating high-dim toy data (p=2000, 2 clusters)...\n")
set.seed(42)
data    <- simu_highdim(n_val, true_p, p_val, mu_val, sigma_val)
X_data  <- scale(data$X, center = TRUE, scale = FALSE)

cat("\n--- Running SCVX Grid Search via run_scvx ---\n")
scvx_best <- run_scvx(X_data, K = 2, seed = 42)

if (!is.na(scvx_best$silhouette)) {
    acc_scvx <- get_cluster_acc(scvx_best$cluster, data$label)
    cat(sprintf("SCVX Selection Outcome:\n"))
    cat(sprintf(" - Best G1: %.2f\n",          scvx_best$g1))
    cat(sprintf(" - Best G2: %.2f\n",          scvx_best$g2))
    cat(sprintf(" - Max Silhouette: %.4f\n",   scvx_best$silhouette))
    cat(sprintf(" - Final Accuracy: %.4f\n",   acc_scvx))
    cat(sprintf(" - Features Selected: %d\n",  sum(scvx_best$selected)))
} else {
    acc_scvx <- NA
    cat("SCVX Grid Search failed to find a valid model.\n")
}

cat("\n--- Final Check ---\n")
if (!is.na(acc_scvx)) {
    cat("SUCCESS: Accuracy is not NA for SCVX Grid Search!\n")
} else {
    cat("FAILURE: SCVX Grid Search returned NA.\n")
}
