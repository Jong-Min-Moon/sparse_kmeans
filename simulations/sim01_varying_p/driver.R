# ---------------------------------------------------------
# 1. Load ALL Libraries (Absolute Priority for S4/CVXR)
# ---------------------------------------------------------
library(methods)
library(MASS)
library(mclust)
library(RSpectra) # Load here to avoid mid-script Matrix conflicts
library(CVXR)
library(cluster) # For Silhouette Index
cat(sprintf("R Version: %s\n", R.version.string))
cat(sprintf("Job Start Time: %s\n", as.character(Sys.time())))

args <- commandArgs(trailingOnly = TRUE)


job_id <- 1
n_iter <- NULL
p_val <- 50 # Default p
out_dir <- "results"

if (length(args) > 0) {
  for (i in seq(1, length(args), by = 2)) {
    arg <- args[i]
    val <- args[i + 1]
    if (arg == "--job_id") {
      job_id <- as.integer(val)
    }
    if (arg == "--n_iter") {
      n_iter <- as.integer(val)
    }
    if (is.null(n_iter)) {
      n_iter <- 1000
    }
    if (arg == "--p") {
      p_val <- as.integer(val)
    }
    if (arg == "--out_dir") {
      out_dir <- val
    }
  }
}

# ---------------------------------------------------------
# Load Source Code
# ---------------------------------------------------------
# Source from ../../code_r/
source("../../code_r/block_coordinate_optim_thompson.R")
source("../../code_r/utils.R")
source("../../code_r/clustering_block_knowncov.R")
source("../../code_r/selection_block_greedy_screening.R")
source("../../code_r/cluster_spectral.R")
source("../../code_r/sdp_kmeans.R")

set.seed(job_id)

# ---------------------------------------------------------
# Data Generation
# ---------------------------------------------------------
n <- 200
p <- p_val
K <- 2
noise_sds <- rep(1, p) # Identity Covariance

# Signal Strength
# mu_j = sqrt(0.4) for j in 1:10
S_0 <- 1:10
m <- sqrt(0.4)

mu1 <- rep(0, p) # Center of cluster 1 (relative to origin? No, paper says +/- mu)
# Actually paper says: X ~ N(mu*, Sigma) and X ~ N(-mu*, Sigma)
# So distance is 2*mu*.
# We set mu* = (m, m, ..., 0, 0)
mu_star <- rep(0, p)
mu_star[S_0] <- m

# Cluster 1: N(mu*, I)
X1 <- matrix(rnorm(n / 2 * p), nrow = n / 2) + matrix(rep(mu_star, n / 2), nrow = n / 2, byrow = TRUE)

# Cluster 2: N(-mu*, I)
X2 <- matrix(rnorm(n / 2 * p), nrow = n / 2) + matrix(rep(-mu_star, n / 2), nrow = n / 2, byrow = TRUE)

# --- Verification of Signal Strength ---
center_diff <- mu_star - (-mu_star)
dist_sq <- sum(center_diff^2)
cat(sprintf("Signal Strength Verification:\n"))
cat(sprintf("||mu* - (-mu*)||^2 = %.4f (Expected: 16.0)\n", dist_sq))
if (abs(dist_sq - 16) > 1e-5) {
  warning("Signal strength deviation detected!")
}
# ---------------------------------------

X <- rbind(X1, X2)
X <- t(X) # p x n
true_labels <- c(rep(1, n / 2), rep(2, n / 2))
# Note: cluster labels 1 and 2 might be swapped in outcome

# ---------------------------------------------------------
# Run Algorithm
# ---------------------------------------------------------
cat(sprintf("Job %d (p=%d): Starting Simulation...\n", job_id, p))
start_time <- Sys.time()

# Known Covariance (Identity)
# Grid Search for C
C_values <- c(0.5)
best_obj <- -Inf
best_res <- NULL
results_all <- list()

# Pre-calculate Distance Matrix for Silhouette (on full data X)
cat("Pre-calculating distance matrix for Silhouette index...\n")
dist_x <- dist(t(X))

for (C_val in C_values) {
  cat(sprintf("  Running for C=%.1f...\n", C_val))

  # Run Algorithm
  res <- block_coordinate_optim_thompson(X, K, n_iter = n_iter, C = C_val, n_perms = 200, covariance = NULL)

  # Calculate Silhouette Index
  # Note: silhouette requires at least 2 clusters to be valid
  if (length(unique(res$cluster)) < 2) {
    cat(sprintf("    Warning: Only 1 cluster found for C=%.1f. Setting silhouette to -1.\n", C_val))
    avg_sil <- -1
  } else {
    sil_res <- silhouette(res$cluster, dist_x)
    avg_sil <- mean(sil_res[, "sil_width"])
  }

  cat(sprintf("  C=%.1f => Avg Silhouette Width: %.4f\n", C_val, avg_sil))

  # Store Result
  results_all[[as.character(C_val)]] <- list(res = res, sil = avg_sil)

  # Update Best based on Silhouette
  if (avg_sil > best_obj) {
    best_obj <- avg_sil # best_obj now stores best silhouette
    best_res <- res
  }
}

res <- best_res

end_time <- Sys.time()
runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
# 1. Clustering Accuracy (Hungarian/Map)
# Simple check for K=2
acc <- max(mean(res$cluster == true_labels), mean(res$cluster != true_labels))

# 2. Variable Selection
# True Active Set: 1:10
selected_indices <- which(res$selected)
tp <- length(intersect(selected_indices, S_0))
fp <- length(setdiff(selected_indices, S_0))
fn <- length(setdiff(S_0, selected_indices))
recall <- tp / length(S_0)
precision <- if (length(selected_indices) > 0) tp / length(selected_indices) else 0
f1 <- if ((precision + recall) > 0) 2 * (precision * recall) / (precision + recall) else 0

output <- list(
  job_id = job_id,
  p = p,
  n = n,
  result = list(
    cluster = res$cluster,
    selected = res$selected,
    alpha = res$alpha,
    beta = res$beta,
    objective = res$objective,
    best_C = C_values[which.max(sapply(results_all, function(r) r$sil))]
  ),
  results_all = results_all, # Save all runs for analysis
  metrics = list(
    accuracy = acc,
    silhouette = best_obj,
    tp = tp,
    fp = fp,
    fn = fn,
    recall = recall,
    precision = precision,
    f1 = f1
  )
)

# Save
dir.create(out_dir, showWarnings = FALSE)
saveRDS(output, file = file.path(out_dir, sprintf("sim_p%d_id%d.rds", p, job_id)))
cat(sprintf("Job %d (p=%d): DONE. Acc=%.4f, TP=%d, FP=%d\n", job_id, p, acc, tp, fp))
