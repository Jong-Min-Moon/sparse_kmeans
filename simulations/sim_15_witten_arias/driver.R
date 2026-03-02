# ---------------------------------------------------------
# Single Simulation Run: sim_15_witten_arias
# ---------------------------------------------------------

# Arguments for HPC
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
source("../../code_r/competitors_modernized.R")
source("../../code_r/get_cluster_acc.R")

# ---------------------------------------------------------
# Data Generation Parameters
# ---------------------------------------------------------
# MATCHING SIM14 SETTINGS with modifications:
# p=400, n=200, K=2, rho=0.45, separation=4 or 5, support=1:10
p <- 400
n <- 200
K <- 2
rho_param <- 45
rho <- rho_param / 100
precision_sparsity <- 2
support <- 1:10
flip <- FALSE

cat(sprintf("--- Simulation Run sim_15_witten_arias (Job ID: %d) ---\n", job_id))
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
# Run Algorithms (Witten and Arias)
# ---------------------------------------------------------
cat("Running Witten's Sparse K-Means...\n")
start_time_witten <- Sys.time()
witten_cluster <- tryCatch(
    {
        run_witten(X, K = K, seed = current_seed)
    },
    error = function(e) {
        cat("Witten failed.\n")
        rep(NA, n)
    }
)
end_time_witten <- Sys.time()
runtime_witten <- as.numeric(difftime(end_time_witten, start_time_witten, units = "secs"))


cat("Running Arias-Castro's Sparse K-Means...\n")
start_time_arias <- Sys.time()
arias_cluster <- tryCatch(
    {
        run_arias(X, K = K, seed = current_seed)
    },
    error = function(e) {
        cat("Arias failed.\n")
        rep(NA, n)
    }
)
end_time_arias <- Sys.time()
runtime_arias <- as.numeric(difftime(end_time_arias, start_time_arias, units = "secs"))


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
cat("\n--- Results ---\n")

acc_witten <- if (all(is.na(witten_cluster))) NA else get_cluster_acc(witten_cluster, true_labels)
acc_arias <- if (all(is.na(arias_cluster))) NA else get_cluster_acc(arias_cluster, true_labels)

cat(sprintf("Witten Runtime: %.2f seconds\n", runtime_witten))
cat(sprintf("Witten Accuracy: %.4f\n", acc_witten))

cat(sprintf("\nArias Runtime: %.2f seconds\n", runtime_arias))
cat(sprintf("Arias Accuracy: %.4f\n", acc_arias))


# Save Result
dir.create("results", showWarnings = FALSE)
saveRDS(list(
    witten = list(
        cluster = witten_cluster,
        acc = acc_witten,
        runtime = runtime_witten
    ),
    arias = list(
        cluster = arias_cluster,
        acc = acc_arias,
        runtime = runtime_arias
    ),
    job_id = job_id,
    params = list(p = p, n = n, rho = rho, sep = separation)
), file = sprintf("results/sim_id%d_sep%d.rds", job_id, separation))

cat("Finished successfully.\n")
