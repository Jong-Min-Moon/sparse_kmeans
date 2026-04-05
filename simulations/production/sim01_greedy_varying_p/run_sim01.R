# Simulation 01 Driver: Greedy Optimization (Single Replicate)
# Intended for SLURM Job Arrays
# Usage: Rscript run_sim01.R --p 1000 --rep 1

library(stats)
library(mclust)
library(Matrix)

# Source functions
source("../../../code_r/sdp_kmeans.R")
source("../../../code_r/utils.R")
source("../../../code_r/selection_block_greedy_screening.R")
source("../../../code_r/clustering_block_knowncov.R")
source("../../../code_r/cluster_greedy.R")

# Use parallel threads for the solver (matching --cpus-per-task=4)
Sys.setenv(OMP_NUM_THREADS = 4)

# Parse arguments manually for better compatibility
args <- commandArgs(trailingOnly = TRUE)

out_dir <- "output"

if (length(args) > 0) {
    for (i in seq(1, length(args), by = 2)) {
        arg <- args[i]
        val <- args[i + 1]
        if (arg == "--p") p <- as.integer(val)
        if (arg == "--rep") rep_id <- as.integer(val)
        if (arg == "--out_dir") out_dir <- val
    }
}

# Fixed Parameters
n <- 200
K <- 2
s <- 10
mu_val <- sqrt(4 / s) # Signal strength

# Create output directory
if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
}

set.seed(2024 + rep_id) # Unique seed per rep

cat(sprintf("Starting Sim01: p=%d, Rep=%d\n", p, rep_id))

# Generate Data
n1 <- n / 2
n2 <- n / 2
mu1 <- numeric(p)
mu2 <- numeric(p)
mu1[1:s] <- mu_val
mu2[1:s] <- -mu_val

X1 <- matrix(rnorm(n1 * p), nrow = p) + mu1
X2 <- matrix(rnorm(n2 * p), nrow = p) + mu2
X <- cbind(X1, X2) # X is p x n

true_labels <- c(rep(1, n1), rep(2, n2))

# Run Algorithm
t_start <- Sys.time()
res <- cluster_greedy(X, K, n_iter = 100, stable_iter = 10, fdr_level = 0.4)
t_end <- Sys.time()
runtime <- as.numeric(difftime(t_end, t_start, units = "secs"))

# Calculate Metrics
ari <- mclust::adjustedRandIndex(res$cluster, true_labels)
# Clustering accuracy for K=2: max of matching or mismatching labels
acc <- max(mean(res$cluster == true_labels), mean(res$cluster != true_labels))

# Prepare Result List (mirroring existing simulation patterns)
output <- list(
    p = p,
    rep = rep_id,
    metrics = list(
        time = runtime,
        ari = ari,
        acc = acc,
        iterations = res$iter
    ),
    result = list(
        cluster = res$cluster,
        iter = res$iter
    )
)

# Save Individual Result
out_file <- file.path(out_dir, sprintf("result_p%d_rep%d.rds", p, rep_id))
saveRDS(output, file = out_file)

cat(sprintf("Completed: Time=%.2fs, ARI=%.4f, ACC=%.4f | Saved to %s\n", runtime, ari, acc, out_file))
