# ---------------------------------------------------------
# Single Simulation Run: sim_19_greedy_naive_bayes
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
library(cluster)

# Arguments for simulation run
args <- commandArgs(trailingOnly = TRUE)

job_id <- 1
separation <- 4
fdr_level <- 0.4
n_iter_greedy <- 100
n_step_admm <- 5000

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
        if (args[i] == "--fdr" && i < length(args)) {
            val <- suppressWarnings(as.numeric(args[i + 1]))
            if (!is.na(val)) fdr_level <- val
        }
        if (args[i] == "--n_step_admm" && i < length(args)) {
            val <- suppressWarnings(as.integer(args[i + 1]))
            if (!is.na(val)) n_step_admm <- val
        }
    }
}

# ---------------------------------------------------------
# Source Code
# ---------------------------------------------------------
# Ensure running from script location
current_dir <- getwd()
relative_path_prefix <- "../../code_r/"

source(paste0(relative_path_prefix, "data_generator.R"))
source(paste0(relative_path_prefix, "block_coordinate_optim_greedy.R"))
source(paste0(relative_path_prefix, "selection_block_greedy_screening.R"))
source(paste0(relative_path_prefix, "clustering_block_knowncov.R"))
source(paste0(relative_path_prefix, "sdp_kmeans.R"))
source(paste0(relative_path_prefix, "utils.R"))
source(paste0(relative_path_prefix, "ESSC.R"))
source(paste0(relative_path_prefix, "get_cluster_acc.R"))

# ---------------------------------------------------------
# Data Generation Parameters
# ---------------------------------------------------------
p <- 400
n <- 200
K <- 2
rho <- 0.45
precision_sparsity <- 2
support <- 1:10
flip <- FALSE

cat(sprintf("--- Simulation Run sim_19_greedy_naive_bayes (Job ID: %d, Sep: %.1f, Target FDR: %.2f) ---\n", job_id, separation, fdr_level))

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

cat("Running Greedy FDR Block Optimization...\n")

start_time <- Sys.time()

res <- tryCatch(
    {
        # Provide `true_labels` explicitly to trace accuracy each iteration
        block_coordinate_optim_greedy(
            X_tilde = X,
            K = K,
            n_iter = n_iter_greedy,
            stable_iter = 10,
            fdr_level = fdr_level,
            max_iter_sdp = n_step_admm,
            true_labels = true_labels
        )
    },
    error = function(e) {
        warning(paste("Greedy Optim Failed:", e$message))
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
empirical_fdr <- NA
power <- NA

if (!is.null(res)) {
    acc <- get_cluster_acc(res$cluster, true_labels)
    cat(sprintf("Balanced Accuracy (Acc): %.4f\n", acc))

    # Robustly handle selected features regardless of format returned
    if (is.null(res$selected)) {
        selected_indices <- integer(0)
    } else if (is.logical(res$selected)) {
        selected_indices <- which(res$selected)
    } else if (is.numeric(res$selected)) {
        # Check if it's 0/1 indicator or actual variable indices
        if (all(res$selected %in% c(0, 1))) {
            selected_indices <- which(res$selected == 1)
        } else {
            selected_indices <- as.integer(res$selected)
        }
    } else {
        selected_indices <- integer(0)
        warning("Unknown format for res$selected, defaulting to empty.")
    }
    
    L <- length(selected_indices)
    
    # Calculate empirical FDR and Power
    true_support <- support
    tp <- sum(selected_indices %in% true_support)
    fp <- length(selected_indices) - tp
    
    # Power max = true_support length
    power <- ifelse(length(true_support) > 0, tp / length(true_support), NA)
    empirical_fdr <- ifelse(L > 0, fp / L, 0)

    cat(sprintf("Number of selected features L: %d\n", L))
    cat(sprintf("True Positives: %d\n", tp))
    cat(sprintf("False Positives: %d\n", fp))
    cat(sprintf("Empirical FDR: %.4f\n", empirical_fdr))
    cat(sprintf("Power: %.4f\n", power))
} else {
    cat("Simulation failed natively or returned degenerated output.\n")
}

# Save Result
dir.create("results_raw", showWarnings = FALSE)
saveRDS(list(
    job_id = job_id,
    sep = separation,
    fdr_level = fdr_level,
    accuracy = acc,
    L = L,
    tp = tp,
    fp = fp,
    power = power,
    empirical_fdr = empirical_fdr,
    runtime = runtime,
    params = list(p = p, n = n, rho = rho, n_iter_greedy = n_iter_greedy, n_step_admm = n_step_admm)
), file = sprintf("results_raw/sim_id%d_sep%d_fdr%s.rds", job_id, separation, format(fdr_level, nsmall = 2)))
