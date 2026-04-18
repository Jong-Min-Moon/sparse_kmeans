# ------------------------------------------------------------------
# sim_laplace.R
# Standalone HPC Array Script: cluster_greedy_ISEE under Laplace Noise
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
job_id <- 1
p <- 100
sep <- 3

if (length(args) > 0) {
    for (i in seq_along(args)) {
        if (args[i] == "--job_id" && i < length(args)) job_id <- as.integer(args[i + 1])
        if (args[i] == "--p" && i < length(args)) p <- as.integer(args[i + 1])
        if (args[i] == "--sep" && i < length(args)) sep <- as.numeric(args[i + 1])
    }
}

noise <- "Laplace"
n <- 500
rho <- 0.45
precision_sparsity <- 2
K <- 2
fdr_target <- 0.4
n_perms <- 5000

cat(sprintf("\n--- HPC Driver: Laplace Job %d, p=%d, sep=%.1f ---\n", job_id, p, sep))

# 1. Source Dependencies directly from code_r
source("../../../code_r/data_generator.R")
source("../../../code_r/cluster_greedy_ISEE.R")
source("../../../code_r/ESSC.R")
source("../../../code_r/ISEE_residual_lasso.R")
source("../../../code_r/get_intercept_residual_lasso.R")
source("../../../code_r/get_cov_small.R")
source("../../../code_r/ISEE_bicluster.R")
source("../../../code_r/clustering_block_knowncov.R")
source("../../../code_r/sdp_kmeans.R")
source("../../../code_r/get_cluster_acc.R")
source("../../../code_r/utils.R")

# 2. Data Generation
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
    n = n,
    seed = 2025 + job_id * 1000 + p,
    noise = noise
)

X <- data_res$X # (p x n) matrix
true_labels <- data_res$labels

# 3. Register Parallel for ISEE
num_cores <- parallel::detectCores() - 1
if (num_cores < 1) num_cores <- 1
if (Sys.getenv("SLURM_CPUS_PER_TASK") != "") {
    num_cores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK"))
}
doParallel::registerDoParallel(cores = min(num_cores, 10))

# 4. Execute cluster_greedy_ISEE
st <- Sys.time()
res <- tryCatch(
    {
        cluster_greedy_ISEE(
            X = X,
            K = K,
            n_iter = 200,
            n_perms = n_perms,
            fdr_target = fdr_target,
            stable_iter = 5,
            true_labels = true_labels
        )
    },
    error = function(e) {
        warning(sprintf("cluster_greedy_ISEE failed: %s", e$message))
        NULL
    }
)
end_time <- Sys.time()
runtime <- as.numeric(difftime(end_time, st, units = "secs"))

# 5. Output Processing
if (!is.null(res)) {
    acc <- get_cluster_acc(res$cluster, true_labels)
    ari <- mclust::adjustedRandIndex(res$cluster, true_labels)

    # Variable Selection metrics
    support <- 1:10
    selected_indices <- res$s_hat
    tp <- length(intersect(selected_indices, support))
    fp <- length(setdiff(selected_indices, support))

    res_df <- data.frame(
        job_id          = job_id,
        p               = p,
        n               = n,
        sep             = sep,
        rho             = rho,
        accuracy        = acc,
        ari             = ari,
        tp              = tp,
        fp              = fp,
        runtime         = runtime,
        n_selected      = length(selected_indices)
    )

    log_msg <- sprintf(
        "[%s] Job %d, p=%d: Acc=%.3f, ARI=%.3f, TP=%d, FP=%d, Runtime=%.1fs\n",
        format(Sys.time(), "%Y-%m-%d %H:%M:%S"), job_id, p, acc, ari, tp, fp, runtime
    )
} else {
    res_df <- data.frame(
        job_id          = job_id,
        p               = p,
        n               = n,
        sep             = sep,
        rho             = rho,
        accuracy        = NA,
        ari             = NA,
        tp              = NA,
        fp              = NA,
        runtime         = runtime,
        n_selected      = NA
    )
    log_msg <- sprintf("[%s] Job %d, p=%d: FAILED\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), job_id, p)
}

# 6. Save Checkpoint
out_dir <- file.path("results_raw", tolower(noise), sprintf("p%d", p))
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_file <- file.path(out_dir, sprintf("sim_job%d_p%d.rds", job_id, p))
saveRDS(res_df, file = out_file)

cat(log_msg)
cat(sprintf("Result saved to %s\n", out_file))
