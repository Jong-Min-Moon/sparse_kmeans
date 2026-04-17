# ==============================================================================
# Simulation Replicate Execution: Thompson Sampling for High-Dimensional Clustering
# ==============================================================================
# Objective:
# This script executes a single replicate of a high-dimensional simulation study
# evaluating the performance of block coordinate optimization with Thompson Sampling
# (`cluster_thompson`). The performance is assessed on a symmetric Gaussian
# mixture model specifically employing an identity covariance structure to isolate
# behavior under isotropic noise conditions (uncorrelated features).
#
# Generative Statistical Model:
# Let n = 200 (Total observations) and p = 5000 (Dimension of feature space).
# K = 2 (Number of true underlying clusters).
# The active feature set defining structural differences is S0 = {1 ... 10}.
# Separation between the two centroids is defined to satisfy: || mu* - (-mu*) ||^2 = 16.
#
# Formally, the data generation implies:
# X ~ N(mu*, I) for Group 1
# X ~ N(-mu*, I) for Group 2
#
# where mu*_j = sqrt(0.4) for j in S0 and 0 otherwise.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Dependency Initialization
# ------------------------------------------------------------------------------
# Loading the requisite mathematical and structural manipulation libraries
library(methods)
library(MASS) # Utilized natively for multivariate normal sampling (mvrnorm)
library(mclust) # Utilized for stability indexing (Adjusted Rand Index capabilities)
library(Matrix) # High-performance sparse mapping matrices
library(foreach) # Parallel loop structuring
library(doParallel) # Local multithread backends
library(glmnet) # Lasso implementation dependencies (Used by downstream methods)
library(kernlab) # Support vector machines and kernel tools
library(matrixStats) # Vectorized row/column execution speeds
library(RSpectra) # Efficient generalized eigenvalue approximations
library(CVXR) # Convex optimization engines
library(cluster) # Clustering methodologies

# ------------------------------------------------------------------------------
# 2. Command Line Arguments & Replicate Indexing
# ------------------------------------------------------------------------------
# Parameters can be dynamically overwritten by Slurm array handlers natively.
args <- commandArgs(trailingOnly = TRUE)

# Default parameter configurations mapped natively if no arguments are provided.
# job_id acts as the unique identifier and seed determinator for the replicate.
separation <- 4 # Targets the condition || mu* - (-mu*) ||^2 = 16 where 4^2 = 16
n_step_admm <- 3000
p <- 5000 # Default High-dimensional structural feature parameter
thompson_step <- 2000
noise <- "Laplace"
# Extract dynamically exported values matching the submission wrapper environment
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
            noise_in <- args[i + 1]
            if (tolower(noise_in) == "laplace") {
                noise <- "Laplace"
            } else if (tolower(noise_in) == "gaussian") {
                noise <- "Gaussian"
            } else {
                noise <- noise_in
            }
        }
    }
}

# ------------------------------------------------------------------------------
# 3. Method Inclusion Mapping
# ------------------------------------------------------------------------------
# Source the core algorithmic implementations from the repository architecture
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
# 4. Global Simulation Parameters Configuration
# ------------------------------------------------------------------------------
n <- 500 # n: Constrained observation limit, mimicking challenging p >> n scenarios
K <- 2 # K: Fixed predefined partition mapping structures
support <- 1:10 # S0: Defining the set of purely informative indices

# Trace executing context gracefully to standard output arrays
cat(sprintf("--- Simulation Run unknowncov_n500_chain_thompson (Noise: %s, Job ID: %d, Sep: %.1f) ---\n", noise, job_id, separation))

# Secure global pseudo-randomization targeting independent array trajectories
set.seed(2026 + job_id)

# ------------------------------------------------------------------------------
# 5. Data Generation Process (Structural Identity Model)
# ------------------------------------------------------------------------------
cat("Instantiating chain graph high-dimensional geometric distributions...\n")

generator_spec <- get_specification_chaingraph(
    support = support,
    separation = separation,
    dimension = p,
    precision_sparsity = 2,
    conditional_correlation = 0.45,
    flip = FALSE
)

# ------------------------------------------------------------------------------
# 5a. Diagnostic Verification (Validation Against Formal Setting)
# ------------------------------------------------------------------------------
# This isolates purely the conceptual difference matrix representing || mu* - (-mu*) ||^2.
# As defined, || mu* - (-mu*) ||^2 = || 2 * mu* ||^2 = 4 * || mu* ||^2
# If |S0| = 10, and mu*_j = sqrt(0.4), ||mu*||^2 = 10 * 0.4 = 4.
# Therefore, 4 * 4 = 16. By definition, separation squared is tracked as 16.
mu1 <- generator_spec$mu1 # Extrapolates to -mu* structurally
mu2 <- generator_spec$mu2 # Extrapolates to +mu* structurally
diff_norm_squared <- sum((mu2 - mu1)^2)

cat(sprintf("[Verification] Computed analytical signal strength || mu* - (-mu*) ||^2: %.5f\n", diff_norm_squared))

if (abs(diff_norm_squared - 16.0) > 1e-4) {
    warning(sprintf("Signal strength mathematical discrepancy detected! Expected 16.0, Found: %.5f", diff_norm_squared))
}

# ------------------------------------------------------------------------------
# 5b. Empirical Extrapolation
# ------------------------------------------------------------------------------
# Generating empirical normal vectors leveraging the validated structural objects
data_res <- generate_data_from_specification(generator_spec, n, seed = 2026 + job_id, noise = noise)
X <- data_res$X
true_labels <- data_res$labels


# ------------------------------------------------------------------------------
# 6. Experimental Clustering Evaluation
# ------------------------------------------------------------------------------
cat("Launching `cluster_thompson` evaluation sequence with grid search over C...\n")
start_time <- Sys.time()

c_values <- c(0.5, 0.4, 0.3)
results_list <- list()
selected_list <- list()

# 6a. Execute cluster_thompson for all C values
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
                covariance = NULL,
                true_cluster = true_labels
            )
        },
        error = function(e) {
            warning(paste("Thompson Clustering implementation experienced a critical crash:", e$message))
            return(NULL) # Fail gracefully mapping output artifacts dynamically downstream
        }
    )

    results_list[[as.character(c_val)]] <- res_temp
    if (!is.null(res_temp)) {
        selected_list[[as.character(c_val)]] <- res_temp$selected
    }
}

# 6b. Merge selected features (Union across all successful C evaluations)
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

# 6c. Recompute silhouette scores using the merged feature set
best_res <- NULL
best_sil <- -Inf
best_C <- NA

cat("\n--- Recomputing Silhouette Scores on Merged Feature Set ---\n")
for (c_val in c_values) {
    res_temp <- results_list[[as.character(c_val)]]

    if (!is.null(res_temp)) {
        # Compute silhouette strictly over the merged feature space union
        if (n_merged_sel > 0 && length(unique(res_temp$cluster)) > 1) {
            dist_mat <- dist(t(X[merged_indices, , drop = FALSE]))
            sil <- cluster::silhouette(res_temp$cluster, dist_mat)
            avg_sil <- mean(sil[, 3])
        } else {
            avg_sil <- -1 # Invalid clustering or no features selected
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
# 7. Post-Hoc Evaluation Metrics & Extraction
# ------------------------------------------------------------------------------
cat("\n=== Diagnostic Simulation Results ===\n")
cat(sprintf("Total evaluation computation cycle mapping: %.2f seconds\n", runtime))

# Default null instantiations tracking failed boundaries
acc <- NA
n_selected <- NA
tp <- NA
fp <- NA
recall <- NA
precision <- NA

if (!is.null(res)) {
    # 7a. Clustering Misclassification Check
    # Mapped by get_cluster_acc resolving permutation assignment invariant matching natively.
    acc <- get_cluster_acc(res$cluster, true_labels)
    cat(sprintf("Classification Accuracy (Relative Match): %.4f\n", acc))

    # 7b. Support Variable Target Intersection (Sparse Recovery Precision)
    selected_indices <- which(res$selected)
    n_selected <- length(selected_indices)

    # Calculate exactly how many identified variables map successfully matching ground truth.
    tp <- length(intersect(selected_indices, support))
    fp <- length(setdiff(selected_indices, support))
    true_active <- length(support)

    # 7c. Methodological Structural Metrics
    recall <- if (true_active > 0) tp / true_active else 0
    precision <- if (n_selected > 0) tp / n_selected else 0

    cat(sprintf("Variables Flagged By Framework: %d (True Pos: %d | False Pos: %d)\n", n_selected, tp, fp))
    cat(sprintf("Observed Precision Vector: %.4f | Recall Target Ratio: %.4f\n", precision, recall))
} else {
    cat("Execution halted preemptively returning invalid non-object states.\n")
}

# ------------------------------------------------------------------------------
# 8. Data Output Handlers
# ------------------------------------------------------------------------------
# Creates the physical output boundary natively handling file mapping hierarchies dynamically
out_dir <- sprintf("results_raw/%s/p%d", tolower(noise), p)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Generate a cohesive list object encapsulating simulation parameters explicitly
output_object <- list(
    job_id = job_id,
    sep = separation,
    pval = pval,
    accuracy = acc, # standardizing to match precision parameter logic tracking acc -> accuracy
    L = n_selected, # matched exactly natively to sim14
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
        noise = noise
    )
)

final_path <- sprintf("%s/sim_id%d_sep%d_pval%s_%s.rds", out_dir, job_id, separation, format(pval, nsmall = 3), tolower(noise))

# Flush binary sequence saving object explicitly tracking for remote extraction mapping
saveRDS(output_object, file = final_path)
cat(sprintf("Results comprehensively saved and mapped against logical identifier -> %s\n", final_path))
