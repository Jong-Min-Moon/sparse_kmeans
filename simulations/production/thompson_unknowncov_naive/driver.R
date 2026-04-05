# ==============================================================================
# Simulation Replicate Execution: Thompson Sampling for High-Dimensional Clustering
# (Unknown Covariance Baseline)
# ==============================================================================

library(methods)
library(MASS)
library(mclust)
library(Matrix)
library(foreach)
library(doParallel)
library(glmnet)
library(kernlab)
library(matrixStats)
library(RSpectra)
library(CVXR)
library(cluster)

# ------------------------------------------------------------------------------
# 1. Command Line Arguments & Replicate Indexing
# ------------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

job_id <- 1
separation <- 6
pval <- 0.01
n_step_admm <- 3000
p <- 5000
thompson_step <- 2000
noise <- "Gaussian"

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
        if (args[i] == "--pval" && i < length(args)) {
            val <- suppressWarnings(as.numeric(args[i + 1]))
            if (!is.na(val)) pval <- val
        }
        if (args[i] == "--n_step_admm" && i < length(args)) {
            val <- suppressWarnings(as.integer(args[i + 1]))
            if (!is.na(val)) n_step_admm <- val
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

# ------------------------------------------------------------------------------
# 2. Method Inclusion Mapping
# ------------------------------------------------------------------------------
source("../../../code_r/data_generator.R")
source("../../../code_r/cluster_thompson.R")
source("../../../code_r/select_greedily.R")
source("../../../code_r/clustering_block_knowncov.R")
source("../../../code_r/sdp_kmeans.R")
source("../../../code_r/utils.R")
source("../../../code_r/ESSC.R")
source("../../../code_r/get_cluster_acc.R")
source("../../../code_r/reward_thompson.R")

# ------------------------------------------------------------------------------
# 3. Global Simulation Parameters Configuration
# ------------------------------------------------------------------------------
n <- 200
K <- 2
support <- 1:10
rho <- 0.45
precision_sparsity <- 2

cat(sprintf("--- Simulation Run thompson_unknowncov_naive (Job ID: %d, Sep: %.1f, Noise: %s) ---\n", job_id, separation, noise))

set.seed(2026 + job_id)

# ------------------------------------------------------------------------------
# 4. Data Generation Process (Unknown Covariance / Chain Graph Model)
# ------------------------------------------------------------------------------
cat("Instantiating unknown-covariance high-dimensional geometric distributions...\n")

generator_spec <- get_specification_chaingraph(
    support = support,
    separation = separation,
    dimension = p,
    precision_sparsity = precision_sparsity,
    conditional_correlation = rho,
    flip = FALSE
)

# Generate dataset based on noise type
data_res <- generate_data_from_specification(generator_spec, n, seed = 2026 + job_id, noise = noise)
X <- data_res$X
true_labels <- data_res$labels

# ------------------------------------------------------------------------------
# 5. Experimental Clustering Evaluation
# ------------------------------------------------------------------------------
cat("Launching `cluster_thompson` evaluation sequence with grid search over C...\n")
start_time <- Sys.time()

c_values <- c(0.5, 0.4, 0.3)
results_list <- list()
selected_list <- list()

for (c_val in c_values) {
    cat(sprintf("\n--- Evaluating C = %.1f ---\n", c_val))
    res_temp <- tryCatch(
        {
            cluster_thompson(
                X = X,
                K = K,
                n_iter = thompson_step,
                C = c_val,
                n_perms = 5000,
                p_val_threshold = pval,
                n_step_admm = n_step_admm,
                covariance = NULL, # Naive algorithm assumes identity or ignores known covariance structure natively
                true_cluster = true_labels
            )
        },
        error = function(e) {
            warning(paste("Thompson Clustering implementation experienced a critical crash:", e$message))
            return(NULL)
        }
    )

    results_list[[as.character(c_val)]] <- res_temp
    if (!is.null(res_temp)) {
        selected_list[[as.character(c_val)]] <- res_temp$selected
    }
}

# Merge selected features (Union)
if (length(selected_list) > 0) {
    merged_selected <- Reduce(`|`, selected_list)
    merged_indices <- which(merged_selected)
    n_merged_sel <- length(merged_indices)
    cat(sprintf("\nTotal unique features selected across all C values: %d\n", n_merged_sel))
} else {
    merged_selected <- rep(FALSE, p)
    merged_indices <- integer(0)
    n_merged_sel <- 0
}

# Recompute silhouette scores using merged feature set
best_res <- NULL
best_sil <- -Inf
best_C <- NA

cat("\n--- Recomputing Silhouette Scores on Merged Feature Set ---\n")
for (c_val in c_values) {
    res_temp <- results_list[[as.character(c_val)]]

    if (!is.null(res_temp)) {
        if (n_merged_sel > 0 && length(unique(res_temp$cluster)) > 1) {
            dist_mat <- dist(t(X[merged_indices, , drop = FALSE]))
            sil <- cluster::silhouette(res_temp$cluster, dist_mat)
            avg_sil <- mean(sil[, 3])
        } else {
            avg_sil <- -1
        }

        cat(sprintf(">>> C = %.1f completed. Silhouette Index: %.4f (Based on %d merged features), acc: %.4f\n", c_val, avg_sil, n_merged_sel, res_temp$acc_history[length(res_temp$acc_history)]))

        if (avg_sil > best_sil) {
            best_sil <- avg_sil
            best_res <- res_temp
            best_C <- c_val
        }
    }
}

res <- best_res
cat(sprintf("\n=== Selected Optimal C = %.1f (Silhouette Index: %.4f) ===\n", best_C, best_sil))

end_time <- Sys.time()
runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))

# ------------------------------------------------------------------------------
# 6. Post-Hoc Evaluation Metrics & Extraction
# ------------------------------------------------------------------------------
cat("\n=== Diagnostic Simulation Results ===\n")
cat(sprintf("Total evaluation computation cycle mapping: %.2f seconds\n", runtime))

acc <- NA
n_selected <- NA
tp <- NA
fp <- NA
recall <- NA
precision <- NA

if (!is.null(res)) {
    acc <- get_cluster_acc(res$cluster, true_labels)
    cat(sprintf("Classification Accuracy (Relative Match): %.4f\n", acc))

    selected_indices <- which(res$selected)
    n_selected <- length(selected_indices)

    tp <- length(intersect(selected_indices, support))
    fp <- length(setdiff(selected_indices, support))
    true_active <- length(support)

    recall <- if (true_active > 0) tp / true_active else 0
    precision <- if (n_selected > 0) tp / n_selected else 0

    cat(sprintf("Variables Flagged By Framework: %d (True Pos: %d | False Pos: %d)\n", n_selected, tp, fp))
    cat(sprintf("Observed Precision Vector: %.4f | Recall Target Ratio: %.4f\n", precision, recall))
} else {
    cat("Execution halted preemptively returning invalid non-object states.\n")
}

# ------------------------------------------------------------------------------
# 7. Data Output Handlers
# ------------------------------------------------------------------------------
out_dir <- sprintf("results_raw/p%d", p)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

output_object <- list(
    job_id = job_id,
    sep = separation,
    pval = pval,
    noise = noise,
    accuracy = acc,
    L = n_selected,
    best_C = best_C,
    best_sil = best_sil,
    runtime = runtime,
    tp = tp,
    fp = fp,
    recall = recall,
    precision = precision,
    params = list(
        p = p,
        n = n,
        separation = separation,
        pval = pval,
        n_step_admm = n_step_admm,
        rho = rho
    )
)

final_path <- sprintf("%s/sim_id%d_sep%d_pval%s.rds", out_dir, job_id, separation, format(pval, nsmall = 3))

saveRDS(output_object, file = final_path)
cat(sprintf("Results comprehensively saved and mapped against logical identifier -> %s\n", final_path))
