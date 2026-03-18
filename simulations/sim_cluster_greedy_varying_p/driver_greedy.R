# ==============================================================================
# Simulation Replicate Execution: Greedy Optimization (cluster_greedy)
# ==============================================================================
# Objective:
# Execute a single replicate of a high-dimensional simulation study for cluster_greedy.
# Uses Isotropic Gaussian Mixture (Identity Covariance) with varying p.
# ==============================================================================

library(methods)
library(mclust)
library(Matrix)

# ------------------------------------------------------------------------------
# 1. Command Line Arguments
# ------------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

job_id <- 1
p <- 5000
fdr <- 0.4
separation <- 4 # sqrt(separation^2) = 4, so ||mu1-mu2||^2 = 16

if (length(args) > 0) {
    for (i in seq_along(args)) {
        if (args[i] == "--job_id" && i < length(args)) {
            val <- suppressWarnings(as.integer(args[i + 1]))
            if (!is.na(val)) job_id <- val
        }
        if (args[i] == "--p" && i < length(args)) {
            val <- suppressWarnings(as.integer(args[i + 1]))
            if (!is.na(val)) p <- val
        }
        if (args[i] == "--fdr" && i < length(args)) {
            val <- suppressWarnings(as.numeric(args[i + 1]))
            if (!is.na(val)) fdr <- val
        }
    }
}

# ------------------------------------------------------------------------------
# 2. Source Dependencies
# ------------------------------------------------------------------------------
source("../../code_r/data_generator.R")
source("../../code_r/cluster_greedy.R")
source("../../code_r/select_greedily.R")
source("../../code_r/clustering_block_knowncov.R")
source("../../code_r/sdp_kmeans.R")
source("../../code_r/utils.R")
source("../../code_r/get_cluster_acc.R")

# ------------------------------------------------------------------------------
# 3. Data Generation
# ------------------------------------------------------------------------------
n <- 200
K <- 2
support <- 1:10

set.seed(2026 + job_id)

cat(sprintf("--- Sim: cluster_greedy | p=%d | Job: %d ---\n", p, job_id))

spec <- get_specification_identity(support, separation, p)
data_res <- generate_data_from_specification(spec, n, seed = 2026 + job_id)
X <- data_res$X
true_labels <- data_res$labels

# ------------------------------------------------------------------------------
# 4. Execution
# ------------------------------------------------------------------------------
cat("Running cluster_greedy...\n")
t_start <- Sys.time()
res <- cluster_greedy(
    X_tilde = X, 
    K = K, 
    n_iter = 100, 
    stable_iter = 10, 
    fdr_level = fdr,
    true_labels = true_labels
)
t_end <- Sys.time()
runtime <- as.numeric(difftime(t_end, t_start, units = "secs"))

# ------------------------------------------------------------------------------
# 5. Metrics
# ------------------------------------------------------------------------------
ari <- mclust::adjustedRandIndex(res$cluster, true_labels)
acc <- get_cluster_acc(res$cluster, true_labels)

selected_indices <- which(res$selected)
n_selected <- length(selected_indices)
tp <- length(intersect(selected_indices, support))
fp <- length(setdiff(selected_indices, support))
recall <- tp / length(support)
precision <- if (n_selected > 0) tp / n_selected else 0

cat(sprintf("Results: ARI=%.4f, ACC=%.4f, TP=%d, FP=%d, Time=%.2fs\n", 
    ari, acc, tp, fp, runtime))

# ------------------------------------------------------------------------------
# 6. Save Results
# ------------------------------------------------------------------------------
out_dir <- sprintf("results_raw/p%d", p)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

output_object <- list(
    job_id = job_id,
    p = p,
    fdr = fdr,
    accuracy = acc,
    ari = ari,
    n_selected = n_selected,
    tp = tp,
    fp = fp,
    recall = recall,
    precision = precision,
    runtime = runtime,
    params = list(
        n = n,
        K = K,
        support = support,
        separation = separation
    )
)

out_file <- file.path(out_dir, sprintf("result_id%d.rds", job_id))
saveRDS(output_object, file = out_file)
cat(sprintf("Saved result to %s\n", out_file))
