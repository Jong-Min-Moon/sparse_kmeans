# ---------------------------------------------------------
# Single Simulation Run: sim14_thompson
# ---------------------------------------------------------
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

# Arguments for HPC
args <- commandArgs(trailingOnly = TRUE)

job_id <- 1
separation <- 4
pval <- 0.01

n_iter_tvs <- 1000
n_perms <- 10000
n_step_admm <- 4000
# Note: User requested n_perms = 10000 and n_step_admm = 4000 for this exact run configuration

if (length(args) > 0) {
    for (i in 1:length(args)) {
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
    }
}

# Source Code
source("../../code_r/data_generator.R")
source("../../code_r/block_coordinate_optim_thompson.R")
source("../../code_r/selection_block_greedy_screening.R")
source("../../code_r/clustering_block_knowncov.R")
source("../../code_r/sdp_kmeans.R")
source("../../code_r/utils.R")
source("../../code_r/ESSC.R")
source("../../code_r/get_cluster_acc.R")

# ---------------------------------------------------------
# Data Generation Parameters
# ---------------------------------------------------------
p <- 400
n <- 200 # User specifically requested sim15 parameters
K <- 2
rho <- 0.45
precision_sparsity <- 2
support <- 1:10
flip <- FALSE

cat(sprintf("--- Simulation Run sim14_thompson (Job ID: %d, Sep: %.1f, P-Val: %.4f) ---\n", job_id, separation, pval))

set.seed(2025 + job_id)

generator <- get_specification_chaingraph(
    support = support,
    separation = separation,
    dimension = p,
    precision_sparsity = precision_sparsity,
    conditional_correlation = rho,
    flip = flip
)

data_res <- generate_data_from_specification(generator, n, seed = 2025 + job_id)
X <- data_res$X
true_labels <- data_res$labels

# Standardize each feature (row-wise scaling) before clustering
X <- t(scale(t(X)))

cat("Running Thompson Block Optimization...\n")

start_time <- Sys.time()

# C parameter logic is detached from threshold dynamically internally now
# Using default C=0.5
res <- tryCatch(
    {
        block_coordinate_optim_thompson(
            X, K,
            n_iter = n_iter_tvs,
            C = 0.5,
            n_perms = n_perms,
            p_val_threshold = pval,
            n_step_admm = n_step_admm,
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

if (!is.null(res)) {
    acc <- get_cluster_acc(res$cluster, true_labels)
    cat(sprintf("Balanced Accuracy (Acc): %.4f\n", acc))

    selected_indices <- which(res$selected)
    L <- length(selected_indices)
    cat(sprintf("Number of selected features L: %d\n", L))
} else {
    cat("Simulation failed natively or returned degenerated output.\n")
}

# Save Result
dir.create("results_raw", showWarnings = FALSE)
saveRDS(list(
    job_id = job_id,
    sep = separation,
    pval = pval,
    accuracy = acc,
    L = L,
    runtime = runtime,
    params = list(p = p, n = n, rho = rho, n_perms = n_perms, n_step_admm = n_step_admm)
), file = sprintf("results_raw/sim_id%d_sep%d_pval%s.rds", job_id, separation, format(pval, nsmall = 3)))
