#!/usr/bin/env Rscript
# ---------------------------------------------------------
# Single Simulation Run: sim_20_er_thompson
# ---------------------------------------------------------
suppressPackageStartupMessages({
    library(methods)
    library(MASS)
    library(mclust)
    library(Matrix)
    library(glmnet)
    library(RSpectra)
    library(cluster)
})

# Default arguments
rep_id <- 1
output_dir <- "output"

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) {
    rep_id <- suppressWarnings(as.integer(args[1]))
    if (is.na(rep_id)) rep_id <- 1
}
if (length(args) >= 2) {
    output_dir <- args[2]
}

if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

# ---------------------------------------------------------
# Source Code
# ---------------------------------------------------------
# Assuming script is run from simulations/sim_20_er_thompson/
source("../../code_r/data_generator.R")
source("../../code_r/cluster_thompson.R")
# The Thompson optimization script depends on multiple block components 
source("../../code_r/selection_block_greedy_screening.R")
source("../../code_r/clustering_block_knowncov.R")
source("../../code_r/sdp_kmeans.R")
source("../../code_r/utils.R")
source("../../code_r/get_cluster_acc.R")

# Load C++ Shared Libraries explicitly from code_r
if (file.exists("../../code_r/proj_simplex.so")) {
    dyn.load("../../code_r/proj_simplex.so")
}
if (file.exists("../../code_r/selection_utils.so")) {
    dyn.load("../../code_r/selection_utils.so")
}

# ---------------------------------------------------------
# Data Generation Parameters
# ---------------------------------------------------------
p <- 200
n <- 200
sep_target <- NULL # Null defaults it to identical scaled mahalanobis
s <- 10
K <- 2
support_true <- 1:s

cat(sprintf("--- Simulation Run sim_20_er_thompson (Rep ID: %d, Sep: %s) ---\n", rep_id, ifelse(is.null(sep_target), "NULL", sep_target)))

set.seed(rep_id)

generator_res <- generate_erdos_renyi_data(
    n = n,
    p = p,
    separation = sep_target,
    s = s
)

X <- generator_res$X
true_labels <- generator_res$labels


# Standardize each feature (row-wise scaling) before clustering
# X is (p x n)
X <- t(scale(t(X)))

cat("Running Thompson Block Optimization...\n")

start_time <- Sys.time()

res <- tryCatch(
    {
        cluster_thompson(
            X, K,
            n_iter = 100,
            C = 0.5,
            n_perms = 1000,        # Same defaults from script
            p_val_threshold = 0.0005,
            n_step_admm = 4000,
            covariance = NULL,
            true_cluster = true_labels
        )
    },
    error = function(e) {
        warning(paste("Thompson Optim Failed:", e$message))
        return(NULL)
    }
)

end_time <- Sys.time()

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
cat("\n--- Results ---\n")
runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat(sprintf("Runtime: %.2f seconds\n", runtime))

acc <- NA
L <- NA
tp <- NA
fp <- NA

if (!is.null(res)) {
    acc <- get_cluster_acc(res$cluster, true_labels)
    cat(sprintf("Balanced Accuracy (Acc): %.4f\n", acc))

    selected_indices <- which(res$selected)
    L <- length(selected_indices)
    
    tp <- sum(selected_indices %in% support_true)
    fp <- L - tp
    
    cat(sprintf("Number of selected features L: %d\n", L))
} else {
    cat("Simulation failed natively or returned degenerated output.\n")
}

# Save Result
output_file <- file.path(output_dir, sprintf("result_rep_%d.rds", rep_id))
saveRDS(list(
    rep_id = rep_id,
    sep = sep_target,
    accuracy = acc,
    L = L,
    tp = tp,
    fp = fp,
    runtime = runtime,
    params = list(p = p, n = n, s = s)
), file = output_file)

cat(sprintf("\nResult saved to %s\n", output_file))
