# ---------------------------------------------------------
# Single Simulation Run: sim_16 (IF-PCA)
# ---------------------------------------------------------

# Arguments
args <- commandArgs(trailingOnly = TRUE)

job_id <- 1
separation <- 4

if (length(args) > 0) {
    for (i in seq(1, length(args), by = 2)) {
        if (args[i] == "--job_id") job_id <- as.integer(args[i + 1])
        if (args[i] == "--sep") separation <- as.numeric(args[i + 1])
    }
}

# Source Code
source("../../code_r/sparse_symmetric_data_generator.R")
source("../../code_r/ifpca.R")
source("../../code_r/get_cluster_acc.R")

# ---------------------------------------------------------
# Data Generation Parameters
# ---------------------------------------------------------
# EXACT MATCH WITH SIM15 SETTINGS:
p <- 400
n <- 200
K <- 2
rho_param <- 45
rho <- rho_param / 100
precision_sparsity <- 2
support <- 1:10
flip <- FALSE

cat(sprintf("--- Simulation Run sim_16 (Job ID: %d) ---\n", job_id))
cat(sprintf("Params: p=%d, n=%d, sep=%.1f, rho=%.2f\n", p, n, separation, rho))

# 1. Initialize Generator
generator <- sparse_symmetric_data_generator(
    support = support,
    separation = separation,
    dimension = p,
    precision_sparsity = precision_sparsity,
    conditional_correlation = rho,
    flip = flip
)

# 2. Generate Data
current_seed <- 2025 + job_id
set.seed(current_seed)
data_res <- generate_data_from_generator(generator, n, seed = current_seed)
X <- data_res$X
true_labels <- data_res$labels

# ---------------------------------------------------------
# Run IF-PCA
# ---------------------------------------------------------
cat("Running IF-PCA Sparse K-Means...\n")
start_time <- Sys.time()

# Ensure X is p x n for IF-PCA (ifpca.R expects features in rows)
if (ncol(X) > nrow(X)) {
    X_ifpca <- t(X)
} else {
    X_ifpca <- X
}

ifpca_res <- tryCatch(
    {
        if_pca(Data = X_ifpca, K = K, rep = 500, nullsimu = TRUE, pvalcut = log(p) / p, kmeansrep = 20, per = 1, seed = current_seed)
    },
    error = function(e) {
        cat(sprintf("IF-PCA failed: %s\n", e$message))
        NULL
    }
)

end_time <- Sys.time()
runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
cat("\n--- Results ---\n")

if (!is.null(ifpca_res)) {
    acc <- get_cluster_acc(ifpca_res$labels, true_labels)
    L <- ifpca_res$L
} else {
    acc <- NA
    L <- NA
}

cat(sprintf("Runtime: %.2f seconds\n", runtime))
cat(sprintf("Accuracy: %.4f\n", acc))
cat(sprintf("Features Selected: %d\n", L))

# Save Result
dir.create("results", showWarnings = FALSE)
saveRDS(list(
    ifpca = list(
        cluster = if (!is.null(ifpca_res)) ifpca_res$labels else rep(NA, n),
        acc = acc,
        L = L,
        selected_features = if (!is.null(ifpca_res)) ifpca_res$selected_features else NA,
        runtime = runtime
    ),
    job_id = job_id,
    params = list(p = p, n = n, rho = rho, sep = separation)
), file = sprintf("results/sim_id%d_sep%d.rds", job_id, separation))

cat("Finished successfully.\n")
