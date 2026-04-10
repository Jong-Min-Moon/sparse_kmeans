# ==============================================================================
# driver_laplace.R
# HPC Array Script: cluster_greedy under Laplace Noise, Chain-Graph Covariance
# ==============================================================================
# Data setting:  chain-graph precision matrix, rho = 0.45, sparsity = 2
#                — identical to competitors_unknowncov (sim_laplace.R)
# Algorithm:     cluster_greedy (known-covariance greedy, identity assumption)
#                — unchanged from knowncov_greedy/driver.R
# ==============================================================================

args <- commandArgs(trailingOnly = TRUE)
job_id <- 1L
p <- 100L
sep <- 6

if (length(args) > 0) {
    for (i in seq_along(args)) {
        if (args[i] == "--job_id" && i < length(args)) job_id <- as.integer(args[i + 1])
        if (args[i] == "--p" && i < length(args)) p <- as.integer(args[i + 1])
        if (args[i] == "--sep" && i < length(args)) sep <- as.numeric(args[i + 1])
    }
}

noise <- "Laplace"
n <- 200L
rho <- 0.45
precision_sparsity <- 2

cat(sprintf(
    "\n--- unknowncov_greedy_naive | Laplace | Job %d, p=%d, sep=%.1f ---\n",
    job_id, p, sep
))

# ------------------------------------------------------------------------------
# 1. Dependencies
# ------------------------------------------------------------------------------
library(methods)
library(Matrix)
library(mclust) # adjustedRandIndex (diagnostic only)

source("../../../code_r/data_generator.R")
source("../../../code_r/sdp_kmeans.R")
source("../../../code_r/utils.R")
source("../../../code_r/select_greedily.R")
source("../../../code_r/clustering_block_knowncov.R")
source("../../../code_r/cluster_greedy.R")
source("../../../code_r/get_cluster_acc.R")

# ------------------------------------------------------------------------------
# 2. Data Generation  (competitors_unknowncov data setting)
# ------------------------------------------------------------------------------
set.seed(2025 + job_id * 1000 + p)

spec <- get_specification_chaingraph(
    support = 1:10,
    separation = sep,
    dimension = p,
    precision_sparsity = precision_sparsity,
    conditional_correlation = rho,
    flip = FALSE
)

data_res <- generate_data_from_specification(
    specification = spec,
    n             = n,
    seed          = 2025 + job_id * 1000 + p,
    noise         = noise
)

# data_res$X is p x n; cluster_greedy expects p x n
X_tilde <- data_res$X
true_labels <- data_res$labels

cat(sprintf("Data generated: p=%d, n=%d, noise=%s, rho=%.2f\n", p, n, noise, rho))

# ------------------------------------------------------------------------------
# 3. Run cluster_greedy  (algorithm unchanged from knowncov_greedy)
# ------------------------------------------------------------------------------
cat("Launching cluster_greedy...\n")
start_time <- Sys.time()

res <- tryCatch(
    cluster_greedy(
        X_tilde     = X_tilde,
        K           = 2L,
        n_iter      = 100,
        stable_iter = 10,
        fdr_level   = 0.4,
        true_labels = true_labels
    ),
    error = function(e) {
        warning(paste("cluster_greedy failed:", e$message))
        NULL
    }
)

runtime <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
cat(sprintf("Runtime: %.2f seconds\n", runtime))

# ------------------------------------------------------------------------------
# 4. Metrics
# ------------------------------------------------------------------------------
support <- spec$support
acc <- NA_real_
n_selected <- NA_integer_
tp <- NA_integer_
fp <- NA_integer_
recall <- NA_real_
precision <- NA_real_

if (!is.null(res)) {
    acc <- get_cluster_acc(res$cluster, true_labels)
    selected_indices <- which(res$selected)
    n_selected <- length(selected_indices)
    tp <- length(intersect(selected_indices, support))
    fp <- length(setdiff(selected_indices, support))
    recall <- if (length(support) > 0) tp / length(support) else 0
    precision <- if (n_selected > 0) tp / n_selected else 0

    cat(sprintf(
        "Accuracy: %.4f | Selected: %d (TP=%d FP=%d) | Recall=%.4f Precision=%.4f\n",
        acc, n_selected, tp, fp, recall, precision
    ))
} else {
    cat("cluster_greedy returned NULL — recording NA metrics.\n")
}

# ------------------------------------------------------------------------------
# 5. Save result
# ------------------------------------------------------------------------------
res_df <- data.frame(
    job_id     = job_id,
    p          = p,
    n          = n,
    sep        = sep,
    rho        = rho,
    noise      = noise,
    accuracy   = acc,
    n_selected = n_selected,
    tp         = tp,
    fp         = fp,
    recall     = recall,
    precision  = precision,
    runtime    = runtime
)

out_dir <- file.path("results_raw", tolower(noise), sprintf("p%d", p))
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_file <- file.path(out_dir, sprintf("sim_job%d_p%d.rds", job_id, p))
saveRDS(res_df, file = out_file)
cat(sprintf("Result saved to %s\n", out_file))
