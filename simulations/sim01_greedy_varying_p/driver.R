# ==============================================================================
# Simulation 01 Driver: Greedy Optimization — Varying Dimension p
# ==============================================================================
# Objective:
#   Executes a single replicate of a high-dimensional simulation study
#   evaluating `cluster_greedy` under an identity covariance structure.
#   Data generated via the unified `data_generator.R` interface shared across
#   all modern simulations (sim_22_thompson_identity_laplace, etc.).
#
# Generative Statistical Model:
#   n = 200 observations, K = 2 clusters, S0 = {1 ... 10} active features.
#   Separation || mu* - (-mu*) ||^2 = sep^2 (default sep = 4 → value 16).
#   Noise distribution configurable: "Gaussian" (default) or "Laplace".
#
# Usage (SLURM job array):
#   Rscript driver.R --job_id $SLURM_ARRAY_TASK_ID --sep 4 --p 5000 --noise Laplace
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Dependencies
# ------------------------------------------------------------------------------
library(methods)
library(mclust)   # adjustedRandIndex (diagnostic logging)
library(Matrix)
library(cluster)  # silhouette (optional diagnostics)

# Core algorithmic sources — relative path from sim directory to repo root
source("../../code_r/data_generator.R")
source("../../code_r/utils.R")
source("../../code_r/sdp_kmeans.R")
source("../../code_r/select_greedily.R")
source("../../code_r/clustering_block_knowncov.R")
source("../../code_r/cluster_greedy.R")
source("../../code_r/get_cluster_acc.R")

# ------------------------------------------------------------------------------
# 2. Command-Line Argument Parsing
# ------------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

# Defaults
job_id    <- 1L
separation <- 4
p         <- 5000L
noise     <- "Laplace"

if (length(args) > 0) {
    for (i in seq_along(args)) {
        if (args[i] == "--job_id" && i < length(args)) {
            val <- suppressWarnings(as.integer(args[i + 1]))
            if (!is.na(val)) job_id <- val
        }
        if (args[i] == "--sep" && i < length(args)) {
            val <- suppressWarnings(as.numeric(args[i + 1]))
            if (!is.na(val)) separation <- val
        }
        if (args[i] == "--p" && i < length(args)) {
            val <- suppressWarnings(as.integer(args[i + 1]))
            if (!is.na(val)) p <- val
        }
        if (args[i] == "--noise" && i < length(args)) {
            noise <- args[i + 1]
        }
    }
}

if (!noise %in% c("Gaussian", "Laplace", "t")) {
    stop(sprintf("Unsupported noise type '%s'. Must be one of: Gaussian, Laplace, t.", noise))
}

# ------------------------------------------------------------------------------
# 3. Fixed Simulation Parameters
# ------------------------------------------------------------------------------
n       <- 200L
K       <- 2L
support <- 1:10   # S0: indices of truly informative features

cat(sprintf(
    "--- Simulation 01 (Greedy, Identity Cov) | Job: %d | sep: %.1f | p: %d | noise: %s ---\n",
    job_id, separation, p, noise
))

# ------------------------------------------------------------------------------
# 4. Seeding
# ------------------------------------------------------------------------------
set.seed(2026 + job_id)

# ------------------------------------------------------------------------------
# 5. Data Generation via Unified Interface
# ------------------------------------------------------------------------------
cat("Building identity covariance specification...\n")

generator_spec <- get_specification_identity(
    support    = support,
    separation = separation,
    dimension  = p
)

# Diagnostic: verify signal strength = sep^2
mu1 <- generator_spec$mu1
mu2 <- generator_spec$mu2
diff_norm_sq <- sum((mu2 - mu1)^2)
cat(sprintf("[Verification] ||mu2 - mu1||^2 = %.5f  (expected: %.5f)\n",
            diff_norm_sq, separation^2))
if (abs(diff_norm_sq - separation^2) > 1e-4) {
    warning(sprintf("Signal strength discrepancy! Expected %.5f, got %.5f",
                    separation^2, diff_norm_sq))
}

cat(sprintf("Generating %s-distributed data (n=%d, p=%d)...\n", noise, n, p))
data_res    <- generate_data_from_specification(generator_spec, n,
                                                seed  = 2026 + job_id,
                                                noise = noise)

# data_res$X is n × p; cluster_greedy expects p × n
X_tilde     <- t(data_res$X)   # p × n
true_labels <- data_res$labels

# ------------------------------------------------------------------------------
# 6. Run cluster_greedy
# ------------------------------------------------------------------------------
cat("Launching cluster_greedy...\n")
start_time <- Sys.time()

res <- tryCatch(
    cluster_greedy(
        X_tilde    = X_tilde,
        K          = K,
        n_iter     = 100,
        stable_iter = 10,
        fdr_level  = 0.4,
        true_labels = true_labels
    ),
    error = function(e) {
        warning(paste("cluster_greedy failed:", e$message))
        return(NULL)
    }
)

end_time <- Sys.time()
runtime  <- as.numeric(difftime(end_time, start_time, units = "secs"))

# ------------------------------------------------------------------------------
# 7. Post-Hoc Evaluation Metrics
# ------------------------------------------------------------------------------
cat("\n=== Evaluation Results ===\n")
cat(sprintf("Runtime: %.2f seconds\n", runtime))

acc       <- NA_real_
n_selected <- NA_integer_
tp        <- NA_integer_
fp        <- NA_integer_
recall    <- NA_real_
precision <- NA_real_

if (!is.null(res)) {
    # Clustering accuracy (Hungarian-algorithm-based, permutation invariant)
    acc <- get_cluster_acc(res$cluster, true_labels)
    cat(sprintf("Clustering Accuracy: %.4f\n", acc))

    # Feature selection recovery
    selected_indices <- which(res$selected)
    n_selected       <- length(selected_indices)
    tp               <- length(intersect(selected_indices, support))
    fp               <- length(setdiff(selected_indices,  support))
    true_active      <- length(support)

    recall    <- if (true_active > 0) tp / true_active else 0
    precision <- if (n_selected > 0)  tp / n_selected  else 0

    cat(sprintf("Selected: %d features  |  TP: %d  FP: %d\n", n_selected, tp, fp))
    cat(sprintf("Recall: %.4f  |  Precision: %.4f\n", recall, precision))
} else {
    cat("cluster_greedy returned NULL — recording NA metrics.\n")
}

# ------------------------------------------------------------------------------
# 8. Output
# ------------------------------------------------------------------------------
out_dir <- sprintf("results_raw/p%d", p)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

output_object <- list(
    job_id    = job_id,
    sep       = separation,
    noise     = noise,
    accuracy  = acc,
    L         = n_selected,
    runtime   = runtime,
    tp        = tp,
    fp        = fp,
    recall    = recall,
    precision = precision,
    params    = list(
        p          = p,
        n          = n,
        separation = separation,
        noise      = noise,
        support    = support
    )
)

final_path <- sprintf("%s/sim_id%d_sep%d_%s.rds", out_dir, job_id, separation, noise)
saveRDS(output_object, file = final_path)
cat(sprintf("Saved -> %s\n", final_path))
